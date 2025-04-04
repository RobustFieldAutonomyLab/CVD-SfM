import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union
import cv2
import h5py
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import pycolmap

from .utils import logger
from .deep_front_end import DISK


def load_image_list(file_path, include_intrinsics=False):
    entries = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            name, data = tokens[0], tokens[1:]
            if include_intrinsics:
                model, width, height, *params = data
                camera = pycolmap.Camera(
                    model=model, width=int(width), height=int(height), params=np.array(params, float)
                )
                entries.append((name, camera))
            else:
                entries.append(name)
    logger.info(f"Loaded {len(entries)} image entries from {file_path.name}")
    return entries

def load_multiple_lists(path_pattern, include_intrinsics=False):
    aggregated = []
    for file in Path(path_pattern.parent).glob(path_pattern.name):
        aggregated.extend(load_image_list(file, include_intrinsics))
    return aggregated


def imread(path, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image if gray else image[:, :, ::-1]


def find_h5_datasets(h5_path):
    dataset_names = set()
    with h5py.File(str(h5_path), 'r', libver='latest') as f:
        f.visititems(lambda _, obj: dataset_names.add(obj.parent.name.strip('/')) if isinstance(obj, h5py.Dataset) else None)
    return list(dataset_names)


def adjust_image_size(image, target_size, method):
    if method.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_' + method[4:].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < target_size[0] or h < target_size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, target_size, interpolation=interp)
    elif method.startswith('pil_'):
        interp = getattr(PIL.Image, method[4:].upper())
        pil_image = PIL.Image.fromarray(image.astype(np.uint8))
        resized = np.asarray(pil_image.resize(target_size, interp), dtype=image.dtype)
    else:
        raise ValueError(f"Unsupported resize method: {method}")
    return resized


class FeatureExtractionDataset(torch.utils.data.Dataset):

    default_settings = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",
    }

    def __init__(self, root: Path, config: Dict, paths: Optional[Union[Path, str, List[str]]] = None):
        self.cfg = SimpleNamespace(**{**self.default_settings, **config})
        self.root_dir = root

        if paths is None:
            all_files = []
            for pattern in self.cfg.globs:
                all_files.extend(glob.glob((root / '**' / pattern).as_posix(), recursive=True))
            if not all_files:
                raise RuntimeError(f"No image files located under: {root}")
            self.file_names = sorted(set(Path(f).relative_to(root).as_posix() for f in all_files))
            logger.info(f"Discovered {len(self.file_names)} image files.")
        else:
            if isinstance(paths, (str, Path)):
                self.file_names = load_multiple_lists(Path(paths))
            elif isinstance(paths, list):
                self.file_names = [str(p) if isinstance(p, Path) else p for p in paths]
            else:
                raise TypeError(f"Invalid paths input type: {type(paths)}")
            for fn in self.file_names:
                if not (root / fn).exists():
                    raise FileNotFoundError(f"Missing image file: {root / fn}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        name = self.file_names[index]
        img = imread(self.root_dir / name, gray=self.cfg.grayscale).astype(np.float32)
        orig_size = img.shape[:2][::-1]  # (W, H)

        if self.cfg.resize_max and (self.cfg.resize_force or max(orig_size) > self.cfg.resize_max):
            scale = self.cfg.resize_max / max(orig_size)
            new_size = tuple(int(round(s * scale)) for s in orig_size)
            img = adjust_image_size(img, new_size, self.cfg.interpolation)

        if self.cfg.grayscale:
            img = img[None]
        else:
            img = img.transpose((2, 0, 1))

        return {
            "image": img / 255.0,
            "original_size": np.array(orig_size),
        }


@torch.no_grad()
def run_disk_extraction(
    image_root: Path,
    output_path: Path,
    use_half: bool = True,
    allow_overwrite: bool = False,
) -> Path:

    logger.info("Initializing DISK feature extractor...")
    config = {"grayscale": False, "resize_max": 1600}
    dataset = FeatureExtractionDataset(image_root, config)
                            
    feature_file = output_path
    feature_file.parent.mkdir(parents=True, exist_ok=True)

    processed = set(find_h5_datasets(feature_file)) if feature_file.exists() and not allow_overwrite else set()
    dataset.file_names = [n for n in dataset.file_names if n not in processed]

    if not dataset.file_names:
        logger.info("All features already extracted. Nothing to do.")
        return feature_file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DISK({"name": "disk", "max_keypoints": 5000}).eval().to(device)

    device = next(model.parameters()).device

    loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=1, pin_memory=True)

    for idx, sample in enumerate(tqdm(loader)):
        name = dataset.file_names[idx]
        image = sample["image"].to(device, non_blocking=True)
        prediction = model({"image": image})
        prediction = {k: v[0].cpu().numpy() for k, v in prediction.items()}

        original_size = sample["original_size"][0].numpy()
        prediction["image_size"] = original_size

        if "keypoints" in prediction:
            net_size = np.array(sample["image"].shape[-2:][::-1])
            scaling = (original_size / net_size).astype(np.float32)
            prediction["keypoints"] = (prediction["keypoints"] + 0.5) * scaling - 0.5
            if "scales" in prediction:
                prediction["scales"] *= scaling.mean()
            prediction["keypoints_uncertainty"] = getattr(model, "detection_noise", 1) * scaling.mean()

        if use_half:
            for k in prediction:
                if prediction[k].dtype == np.float32:
                    prediction[k] = prediction[k].astype(np.float16)

        with h5py.File(str(feature_file), 'a', libver='latest') as f:
            try:
                if name in f:
                    del f[name]
                group = f.create_group(name)
                for k, v in prediction.items():
                    group.create_dataset(k, data=v)
                if "keypoints" in prediction:
                    group["keypoints"].attrs["uncertainty"] = prediction["keypoints_uncertainty"]
            except OSError as e:
                if "No space left on device" in str(e):
                    logger.error("Disk full. Consider using --as_half to save space.")
                raise e

        del prediction

    logger.info(f"Feature extraction complete. Data saved to {feature_file}")
    return feature_file