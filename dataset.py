import csv
import os.path as osp
import torch
from torchvision.datasets.folder import pil_loader
import torchvision
import numpy as np
from functools import partial


def create_FFHQ_AGING(data_root, label_file):
    age_clusters = ['0-2', '3-6', '7-9', '10-14', '15-19',
                    '20-29', '30-39', '40-49', '50-69', '70-120']

    f = open(label_file, 'r', newline='')
    reader = csv.DictReader(f)
    img_list = []

    for csv_row in reader:
        age, age_conf = csv_row['age_group'], float(csv_row['age_group_confidence'])
        gender, gender_conf = csv_row['gender'], float(csv_row['gender_confidence'])
        head_pitch, head_roll, head_yaw = float(csv_row['head_pitch']), float(csv_row['head_roll']), float(
            csv_row['head_yaw'])
        left_eye_occluded, right_eye_occluded = float(csv_row['left_eye_occluded']), float(
            csv_row['right_eye_occluded'])
        glasses = csv_row['glasses']

        no_attributes_found = head_pitch == -1 and head_roll == -1 and head_yaw == -1 and \
                              left_eye_occluded == -1 and right_eye_occluded == -1 and glasses == -1

        age_cond = age_conf > 0.6
        gender_cond = gender_conf > 0.66

        head_pose_cond = abs(head_pitch) < 30.0 and abs(head_yaw) < 40.0
        eyes_cond = (left_eye_occluded < 90.0 and right_eye_occluded < 50.0) or (
                left_eye_occluded < 50.0 and right_eye_occluded < 90.0)
        glasses_cond = glasses != 'Dark'

        valid1 = age_cond and gender_cond and no_attributes_found
        valid2 = age_cond and gender_cond and head_pose_cond and eyes_cond and glasses_cond

        if (valid1 or valid2):
            num = int(csv_row['image_number'])
            img_filename = str(num).zfill(5) + '.png'
            glasses = 0 if glasses == 'None' else 1
            gender = 1 if gender == 'male' else 0
            age = age_clusters.index(age)
            img_list.append([osp.join(data_root, img_filename), glasses, age, gender])
    return np.array(img_list)


class FFHQ(torch.utils.data.Dataset):
    def __init__(self, data_root, label_file, transform=None):
        '''
        FFHQ(data_root='/home/zzhuang/DATASET/FFHQ/images256x256',
             label_file='/home/zzhuang/DATASET/FFHQ/ffhq_aging_labels.csv',
             transform=transform)
        :param data_root:
        :param label_file:
        :param transform:
        '''
        self.transform = transform
        self.img_list = create_FFHQ_AGING(data_root, label_file)

    def __getitem__(self, idx):
        line = self.img_list[idx]
        img = pil_loader(line[0])
        if self.transform is not None:
            img = self.transform(img)
        labels = line[1:].astype(int)
        return img, labels

    def __len__(self):
        return len(self.img_list)


dataset_dict = {
    'FFHQ': partial(FFHQ,
                    data_root='/home/zzhuang/DATASET/FFHQ/images256x256',
                    label_file='/home/zzhuang/DATASET/FFHQ/ffhq_aging_labels.csv'),
    'CELEBA': partial(torchvision.datasets.CelebA,
                      root='/home/zzhuang/DATASET',
                      split='all', target_type='attr')
}
