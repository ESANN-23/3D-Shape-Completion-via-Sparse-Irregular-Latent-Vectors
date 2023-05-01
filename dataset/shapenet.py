import torch
from torch.utils import data

import numpy as np

from utils.xgutils import *

import os

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9,
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}

from clip.clip import clip

_, preprocess = clip.load("ViT-B/32")

class VirtualScanSelector():
    def __init__(self, radius=10, context_N=1024, noise=0., manual_cameras={}):
        self.__dict__.update(locals())

    def __call__(self, Xbd, index=None, **kwargs):
        # if index in self.manual_cameras:
        # C = self.manual_cameras[179]
        # else:
        C = geoutil.sample_sphere(1)[0] * self.radius
        Xct = geoutil.hidden_point_removal(Xbd, C)
        if Xct.shape[0] <= 2:
            print("warning, virtual scanned points less than 2")
            print("Use Xbd as Xct")
            Xct = Xbd

        if self.context_N >= 0:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        if self.noise > 0:
            Xct += np.random.randn(*Xct.shape) * self.noise
            Xct = Xct.clip(-1., 1.)
        return Xct

class ShapeNet(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None, sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.partial_selector = VirtualScanSelector()

        if categories is None:
            categories = os.listdir(self.dataset_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.dataset_folder, c)) and c.startswith('0')]
        categories.sort()
        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.dataset_folder, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')

            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        self.length = len(self.models)

    def __getitem__(self, idx):

        index = idx % self.length
        o_ind = index

        category = self.models[idx]['category']
        model = self.models[idx]['model']

        point_path = os.path.join(self.dataset_folder, category, '4_points', model + '.npz')
        try:
            with np.load(point_path) as data:  ### 这里的near_points和voxel_points的数量应该是一样的
                vol_points = data['vol_points']
                vol_label = data['vol_labels']
                near_points = data['near_points']
                near_label = data['near_labels']

                vol_label = np.unpackbits(vol_label)[:vol_points.shape[0]]
                near_label = np.unpackbits(near_label)[:near_points.shape[0]]

        except Exception as e:
            print(e)
            print(point_path)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            pc_path = os.path.join(self.dataset_folder, category, '4_pointcloud', model + '.npz')
            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                point_cloud = surface

            if self.surface_sampling:
                ind = np.random.default_rng().choice(point_cloud.shape[0], self.pc_size, replace=False)
                surface = point_cloud[ind]
            surface = torch.from_numpy(surface)
            Xct = self.get_partial(point_cloud, index=o_ind)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)  ### 这里是volume中的点和surface附近的点拼接之后应为2048
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

        if self.return_surface:
            return points, labels, surface, Xct
        else:
            return points, labels, Xct

    def get_partial(self, Xbd, Xtg=None, Ytg=None, index=None):

        Xct = self.partial_selector(Xbd, index=index)   #### 这里的index是啥？？
        return Xct

    def __len__(self):
        return len(self.models)


if __name__ == '__main__':
    pass