import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from . import utils

resized_h = 480
resized_w = 270

class SingleSatGrdDataset(Dataset):

    def __init__(self,
                 root_dir,
                 transform=None,
                 shift_range_lat=20,
                 shift_range_lon=20,
                 rotation_range=10):
        
        super().__init__()

        self.root_dir = root_dir

        images_dir = os.path.join(root_dir, 'images')
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        all_imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
        if len(all_imgs) == 0:
            raise FileNotFoundError(f"No imagery in {images_dir}")
        self.file_name_list = sorted(all_imgs)
        self.images_dir = images_dir

        sat_dir = os.path.join(root_dir, 'sat')
        sat_files = [f for f in os.listdir(sat_dir) if f.lower().endswith(valid_ext)]
        if len(sat_files) == 0:
            raise FileNotFoundError(f"No imagery in {sat_dir}")
        sat_path = os.path.join(sat_dir, sat_files[0])
        self.sat_map = Image.open(sat_path).convert('RGB')

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel
        self.rotation_range = rotation_range

        if transform is not None:
            self.satmap_transform = transform[0]
            self.queryimage_transform = transform[1]
        else:
            self.satmap_transform = None
            self.queryimage_transform = None

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):

        file_name = self.file_name_list[idx] 

        query_img_path = os.path.join(self.images_dir, file_name)

        with Image.open(query_img_path).convert('RGB') as img:

            query_img = img
            w, h = query_img.size

            ratio = resized_w / w
            
            fx = w * ratio
            fy = h * ratio
            cx = fx / 2 
            cy = fy / 2
            camera_k = torch.tensor([[fx, 0,  cx],
                                          [0,  fy, cy],
                                          [0,  0,   1]], dtype=torch.float32)
            
            if self.queryimage_transform is not None:
                query_img = self.queryimage_transform(img)
            else:
                query_img = TF.to_tensor(img)

        sat_img = self.sat_map.copy()

        camera_shift = utils.CameraGPS_shift_left

        sat_align_cam = sat_img.transform(
            sat_img.size,
            Image.AFFINE,
            (1, 0, camera_shift[0] / self.meter_per_pixel,
             0, 1, camera_shift[1] / self.meter_per_pixel),
            resample=Image.BILINEAR
        )

        gt_shift_x = np.random.uniform(-1, 1)
        gt_shift_y = np.random.uniform(-1, 1)
        sat_rand_shift = sat_align_cam.transform(
            sat_align_cam.size,
            Image.AFFINE,
            (1, 0, gt_shift_x * self.shift_range_pixels_lon,
             0, 1, -gt_shift_y * self.shift_range_pixels_lat),
            resample=Image.BILINEAR
        )

        theta = np.random.uniform(-1, 1)
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(theta * self.rotation_range)

        sat_map = TF.center_crop(
            sat_rand_shift_rand_rot,
            utils.SatMap_process_sidelength
        )

        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        shift_x = torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1)
        shift_y = torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1)
        theta_t = torch.tensor(theta,       dtype=torch.float32).reshape(1)

        return (
            sat_map,
            camera_k,
            query_img,
            shift_x,
            shift_y,
            theta_t,
            file_name
        )

def load_data(batch_size, root_dir, shift_range_lat=20, shift_range_lon=20, rotation_range=10):

    satmap_transform = transforms.Compose([
        transforms.Resize((utils.SatMap_process_sidelength, utils.SatMap_process_sidelength)),
        transforms.ToTensor(),
    ])
    queryimage_transform = transforms.Compose([
        transforms.Resize((resized_h, resized_w)),
        transforms.ToTensor(),
    ])

    dataset = SingleSatGrdDataset(
        root_dir=root_dir,
        transform=(satmap_transform, queryimage_transform),
        shift_range_lat=shift_range_lat,
        shift_range_lon=shift_range_lon,
        rotation_range=rotation_range
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    return data_loader