import tqdm
import numpy as np
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial

from dataset import dataset_dict
from flow.modules import Flow, ActNorm, AdditiveCoupling, InvConv2dLU, ActNormIdentity, Permute2d
from flow.ops import compute_z_shapes, sample_z
from utils import LoggerX, load_network

'''
python train.py \
    --batch_size 16 \
    --max_iter 1000000 \
    --n_flow 32 \
    --n_block 4 \
    --n_bits 5 \
    --lr 1e-4 \
    --img_size 64 \
    --temp 0.7 \
    --gradient_accumulation_steps 1 \
    --checkpoint_iter 100000 \
    --generation_iter 100 \
    --dataset_name FFHQ \
    --network_config 1 \
    --cls_loss_weight 0.001
'''


class FLOW(object):
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root='./output/{}_{}'.format(opt.img_size, opt.network_config))
        self.build_dataloader()
        self.init_module()

    def init_module(self):
        opt = self.opt
        network_configs = [
            {'norm_layer': ActNorm, 'coupling_layer': AdditiveCoupling, 'permute_layer': InvConv2dLU},
            {'norm_layer': ActNormIdentity, 'coupling_layer': AdditiveCoupling,
             'permute_layer': partial(Permute2d, shuffle=True)},
        ]

        dim_condition = 0
        if opt.dataset_name == 'FFHQ':
            dim_condition = 12
        elif opt.dataset_name == 'CELEBA':
            dim_condition = 40

        generator = Flow(
            img_size=opt.img_size,
            in_channel=3,
            n_flow=opt.n_flow,
            n_block=opt.n_block,
            dim_condition=0,
            n_bits=5,
            **network_configs[opt.network_config]
        )
        if opt.pretrained_path is not None:
            generator.load_state_dict(load_network(opt.pretrained_path))
        z_shapes = compute_z_shapes(image_size=opt.img_size,
                                    in_channels=3,
                                    n_stages=opt.n_block)

        y_classifier = nn.Sequential(
            nn.Linear(np.prod(z_shapes[-1]), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim_condition),
        )

        optimizer = optim.Adam(
            list(generator.parameters()) + list(y_classifier.parameters()), lr=opt.lr)

        generator.cuda()
        y_classifier.cuda()

        # local_rank = dist.get_rank()
        # generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[local_rank],
        #                                                       output_device=local_rank)
        # y_classifier = torch.nn.parallel.DistributedDataParallel(y_classifier, device_ids=[local_rank],
        #                                                          output_device=local_rank)
        generator = nn.DataParallel(generator)
        y_classifier = nn.DataParallel(y_classifier)

        self.logger.modules = [generator, y_classifier, optimizer]

        self.generator = generator
        self.y_classifier = y_classifier
        self.optimizer = optimizer

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser(description="Glow trainer")
        parser.add_argument("--batch_size", default=16, type=int, help="batch size")
        parser.add_argument("--max_iter", default=800000, type=int, help="maximum iterations")
        parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
        parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
        parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
        parser.add_argument("--gradient_accumulation_steps", default=4, type=int,
                            help="number of gradient_accumulation")

        parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
        parser.add_argument("--img_size", default=64, type=int, help="image size")
        parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
        parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
        parser.add_argument("--checkpoint_iter", default=1000, type=int)
        parser.add_argument("--generation_iter", default=100, type=int)
        parser.add_argument("--num_workers", default=16, type=int)
        parser.add_argument("--network_config", default=0, type=int)
        parser.add_argument("--cls_loss_weight", default=0.0, type=float)
        parser.add_argument("--pretrained_path", type=str)
        parser.add_argument("--dataset_name", type=str)
        parser.add_argument("--local_rank", default=0, type=int)
        return parser

    def build_dataloader(self):
        opt = self.opt
        if opt.dataset_name == 'CELEBA':
            import warnings
            warnings.warn('celeba dataset is not in square shape')

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([opt.img_size, opt.img_size]),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]
        )
        dataset = dataset_dict[opt.dataset_name](transform=transform)
        # sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        sampler = None

        loader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers,
                            sampler=sampler,
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True)
        self.loader = loader
        self.sampler = sampler

    def calc_loss(self, log_p, logdet):
        opt = self.opt
        n_pixel = opt.img_size * opt.img_size * 3

        loss = -opt.n_bits * n_pixel
        loss = loss + logdet + log_p

        return (
            (-loss / (np.log(2) * n_pixel)).mean(),
            (log_p / (np.log(2) * n_pixel)).mean(),
            (logdet / (np.log(2) * n_pixel)).mean(),
        )

    def fit(self):
        opt = self.opt
        # training routine
        self.progress_bar = tqdm.tqdm(total=opt.max_iter, disable=(dist.get_rank() != 0))

        n_iter = 1
        self.progress_bar.update(n_iter)

        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(n_iter // len(self.loader))
            for inputs in self.loader:
                self.train(inputs, n_iter)
                self.progress_bar.refresh()
                self.progress_bar.update()
                self.post_task(n_iter)

                n_iter += 1
            if n_iter > opt.max_iter:
                break

    def train(self, inputs, n_iter):
        opt = self.opt
        image, labels = inputs[0].cuda(), inputs[1].cuda()

        log_p, log_det, z_list = self.generator(input=image, condition=None, continuous=True)

        nll_loss, log_p, log_det = self.calc_loss(log_p, log_det)

        y_logits = self.y_classifier(torch.flatten(z_list[-1], start_dim=1))

        if opt.dataset_name == 'FFHQ':
            glasses, age, gender = labels[:, 0].unsqueeze(1), labels[:, 1], labels[:, 2].unsqueeze(1)
            glasses_logits, age_logits, gender_logits = y_logits.split([1, 10, 1], dim=1)
            cls_loss = F.binary_cross_entropy_with_logits(glasses_logits, glasses.float()) + \
                       F.binary_cross_entropy_with_logits(gender_logits, gender.float()) + \
                       F.cross_entropy(age_logits, age)
        elif opt.dataset_name == 'CELEBA':
            cls_loss = F.binary_cross_entropy_with_logits(y_logits, labels.float())
        else:
            raise NotImplementedError

        loss = nll_loss + cls_loss * opt.cls_loss_weight

        acc_loss = loss / opt.gradient_accumulation_steps
        acc_loss.backward()

        max_grad_clip = 5
        max_grad_norm = 100
        params = list(self.generator.parameters()) + list(self.y_classifier.parameters())
        torch.nn.utils.clip_grad_value_(params, max_grad_clip)
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        if n_iter % opt.gradient_accumulation_steps == 0:
            if opt.network_config == 0 and n_iter == 1:
                return
            self.optimizer.step()
            self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.msg([nll_loss, cls_loss, log_p, log_det, lr], n_iter)

    @torch.no_grad()
    def post_task(self, n_iter):
        opt = self.opt
        if n_iter % opt.generation_iter == 0:
            z_samples = sample_z(input_shape=(opt.n_sample, 3, opt.img_size, opt.img_size),
                                 n_stages=opt.n_block, temp=opt.temp)
            z_samples = [x.cuda() for x in z_samples]
            gen_images = self.generator.module.reverse(z_list=z_samples, condition=None)
            grid_img = torchvision.utils.make_grid(gen_images, normalize=True)
            self.logger.save_image(grid_img, n_iter, 'train')

        if n_iter % opt.checkpoint_iter == 0:
            self.logger.checkpoints(n_iter)
