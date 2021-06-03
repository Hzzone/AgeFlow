import torch
import torch.distributed as dist

from flow.models import FLOW

if __name__ == '__main__':
    opt = FLOW.build_options().parse_args()
    print(opt)
    # dist.init_process_group(backend='nccl',
    #                         init_method='env://')
    torch.cuda.set_device(opt.local_rank)
    model = FLOW(opt)
    model.fit()
