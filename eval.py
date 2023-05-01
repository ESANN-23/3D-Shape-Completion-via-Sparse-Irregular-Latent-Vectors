import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vqvae_512_1024_2048', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument('--pth', default='None', type=str, help='CheckPoint Path')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_path', default='None', type=str,
                    help='dataset path')
args = parser.parse_args()

import torch
import torch.backends.cudnn as cudnn

import numpy as np

from scipy.spatial import cKDTree as KDTree

import trimesh
import mcubes

from dataset.test_shapenet import ShapeNet, category_ids
from timm.models import create_model
import utils

import sys
sys.path.append("..")

from chamfer_distance import ChamferDistance
# from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

from utils.xgutils import *


def pc_metrics(p1, p2, space_ext=2, fscore_param=0.01, scale=.5):
    """ p2: reference ponits
        (B, N, 3)
    """
    p1, p2, space_ext = p1 * scale, p2 * scale, space_ext * scale
    f_thresh = space_ext * fscore_param

    d1, d2 = chamfer_dist(ptutil.np2th(p1), ptutil.np2th(p2))
    d1sqrt, d2sqrt = (d1 ** .5), (d2 ** .5)
    chamfer_L1 = d1sqrt.mean(axis=-1) + d2sqrt.mean(axis=-1)  # 也是用CD L1算的
    chamfer_L2 = d1.mean(axis=-1) + d2.mean(axis=-1)
    precision = (d1sqrt < f_thresh).sum(axis=-1).float() / p1.shape[1]
    recall = (d2sqrt < f_thresh).sum(axis=-1).float() / p2.shape[1]
    fscore = 2 * torch.div(recall * precision, recall + precision)
    fscore[fscore == float("inf")] = 0

    return chamfer_L1, fscore


def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = create_model(
        args.model,
    )
    device = torch.device(args.device)

    model.eval()
    model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)
    model.to(device)


    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()

    # gtgrid = nputil.makeGrid([-1, -1, -1.], [1., 1, 1], [64, ] * 3, indexing="ij")
    # grid = torch.from_numpy(nputil.makeGrid([-1, -1, -1.], [1., 1, 1], [128, ] * 3, indexing="ij").astype(np.float32))[None, ...].cuda()

    with torch.no_grad():
        
        metric_loggers = []
        for category, category_id in category_ids.items():
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_loggers.append(metric_logger)
            header = 'Test:'

            dataset_test = ShapeNet(args.data_path, split='test', categories=[category_id], transform=None, sampling=False, return_surface=True, surface_sampling=False)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=1,
                num_workers=12,
                drop_last=False,
            )

            for batch in metric_logger.log_every(data_loader_test, 10, header):
                points, labels, surface, _ = batch

                ind = np.random.default_rng().choice(surface[0].numpy().shape[0], 2048, replace=False)

                surface2048 = surface[0][ind][None]

                surface2048 = surface2048.to(device, non_blocking=True)
                points = points.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                N = 5000

                _, latents, centers_quantized, _, _, encodings = model.encode(surface2048)
                centers = centers_quantized.float() / 255.0 * 2 - 1

                output = torch.cat([model.decoder(latents, centers, points[:, i*N:(i+1)*N])[0] for i in range(math.ceil(grid.shape[1]/N))], dim=1)

                pred = torch.zeros_like(output[0])
                pred[output[0]>=0] = 1
                intersection = (pred * labels[0]).sum()
                union = (pred + labels[0]).gt(0).sum()
                iou = intersection * 1.0 / union

                metric_logger.update(iou=iou.item())

                output = torch.cat([model.decoder(latents, centers, grid[:, i*N:(i+1)*N])[0] for i in range(math.ceil(grid.shape[1]/N))], dim=1)



                # output = torch.sigmoid(output)
                # occ = output.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
                # vert, face = geoutil.array2mesh(occ, thresh=.5, coords=grid)
                # # gtvert, gtface = geoutil.array2mesh(labels[0], thresh=.5, coords=gtgrid)
                # # Xbd = loaded["batch"]["Xbd"]
                # pred = geoutil.sampleMesh(vert, face, sampleN=10 ** 5)[None, ...]
                # # gt = geoutil.sampleMesh(gtvert, gtface, sampleN=10 ** 5)[None, ...]
                #
                # # cd, fscore = pc_metrics(recon, gt)
                # # metric_logger.update(cd=cd)
                # # metric_logger.update(fscore=fscore)

                # output = torch.sigmoid(output)
                volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                verts, faces = mcubes.marching_cubes(volume, 0.5)
                verts *= gap
                verts -= 1.
                m = trimesh.Trimesh(verts, faces)

                volume_gt = labels.view(64, 64, 64).cpu().numpy()
                verts_gt, faces_gt = mcubes.marching_cubes(volume_gt, 0)
                verts_gt *= 2./64
                verts_gt -= 1.
                m_gt = trimesh.Trimesh(verts_gt, faces_gt)

                pred = m.sample(100000)
                gt = m_gt.sample(100000)

                tree = KDTree(pred)
                dist, _ = tree.query(surface[0].cpu().numpy())
                d1 = dist
                gt_to_gen_chamfer = np.mean(dist)
                gt_to_gen_chamfer_sq = np.mean(np.square(dist))

                tree = KDTree(surface[0].cpu().numpy())
                dist, _ = tree.query(pred)
                d2 = dist
                gen_to_gt_chamfer = np.mean(dist)
                gen_to_gt_chamfer_sq = np.mean(np.square(dist))

                cd = gt_to_gen_chamfer + gen_to_gt_chamfer

                metric_logger.update(cd=cd)

                th = 0.02

                if len(d1) and len(d2):
                    recall = float(sum(d < th for d in d2)) / float(len(d2))
                    precision = float(sum(d < th for d in d1)) / float(len(d1))

                    if recall+precision > 0:
                        fscore = 2 * recall * precision / (recall + precision)
                    else:
                        fscore = 0
                metric_logger.update(fscore=fscore)

            print(category, metric_logger.iou.avg, metric_logger.cd.avg, metric_logger.fscore.avg)

        print(args)
        for (category, _), metric_logger in zip(category_ids.items(), metric_loggers):
            print(category, metric_logger.iou.avg, metric_logger.cd.avg, metric_logger.fscore.avg)

if __name__ == '__main__':
    main()