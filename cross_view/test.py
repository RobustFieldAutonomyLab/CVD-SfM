import os
import torch
import torch.optim as optim
from dataset import load_data
from model.models import Model
import ssl
import numpy as np
import os
import argparse

ssl._create_default_https_context = ssl._create_unverified_context

def test(net_test, args, save_path, epoch):

    net_test.eval()

    dataloader = load_data(mini_batch, args.root_dir, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    pred_lons = []
    pred_lats = []
    pred_oriens = []
    
    gt_lons = []
    gt_lats = []
    gt_oriens = []

    file_names_all = []
    
    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):

            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in data[:-1]]
            file_names = data[-1]

            if args.proj == 'CrossAttn':
                pred_u, pred_v, pred_orien = net_test.CVattn_rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')
            else:
                pred_u, pred_v, pred_orien = net_test.rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')

            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy())

            file_names_all.extend(file_names)

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)

            if i % 20 == 0:
                print(i)

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)
    
    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    output_file = os.path.join(save_path, 'cross_estimation.csv')

    with open(output_file, 'w') as f:
        f.write("File_Name, Pred_Lon, Pred_Lat, Pred_Orien\n")
        for file_name, pred_lon, pred_lat, pred_orien in zip(file_names_all, pred_lons, pred_lats, pred_oriens):
            f.write(f"{file_name}, {pred_lon}, {pred_lat}, {pred_orien}\n")

    net_test.train()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=2, help='any integer')
    parser.add_argument('--Optimizer', type=str, default='TransV1G2SP', help='')
    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, CrossAttn')
    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')
    parser.add_argument('--root_dir', type=str, help='Root directory for the dataset')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = '.'
    net = Model(args)
    net.to(device)

    net.load_state_dict((torch.load('./model/model.pth')), strict=False)

    test(net, args, save_path, epoch=0)

