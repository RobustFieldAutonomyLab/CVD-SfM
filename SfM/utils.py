import cv2
import h5py
import numpy as np
import pycolmap
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import defaultdict
import logging

logger = logging.getLogger('SfM')
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_image(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """Read image from disk using OpenCV."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise IOError(f"Failed to read image: {path}")
    return image if grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_image_names_from_h5(h5_path: Path) -> List[str]:
    """Get all unique image group names (i.e., image names) from an HDF5 file."""
    names = set()

    def visitor(_, obj):
        if isinstance(obj, h5py.Dataset):
            names.add(obj.parent.name.strip("/"))

    with h5py.File(str(h5_path), "r") as hfile:
        hfile.visititems(visitor)

    return list(names)

def load_keypoints(
    feature_file: Path, image_name: str, with_uncertainty: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    with h5py.File(str(feature_file), "r") as f:
        kpts = f[image_name]["keypoints"][()]
        uncertainty = f[image_name]["keypoints"].attrs.get("uncertainty", None)
    return (kpts, uncertainty) if with_uncertainty else kpts


def read_pairs_from_file(path: Path) -> Dict[str, List[str]]:
    """Load query-reference image pairs from a retrieval file."""
    pair_dict = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q, r = line.split()
            pair_dict[q].append(r)
    return dict(pair_dict)


def parse_image_list_file(path: Path, include_intrinsics: bool = False) -> Union[List[str], List[Tuple[str, pycolmap.Camera]]]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            image_name = tokens[0]
            if include_intrinsics:
                model, width, height, *params = tokens[1:]
                cam = pycolmap.Camera(
                    model=model,
                    width=int(width),
                    height=int(height),
                    params=np.array(params, dtype=np.float64),
                )
                entries.append((image_name, cam))
            else:
                entries.append(image_name)
    logger.info(f"Loaded {len(entries)} images from {path.name}")
    return entries


def gather_image_list_from_pattern(pattern_path: Path, with_intrinsics: bool = False) -> List:
    all_entries = []
    for file in Path(pattern_path.parent).glob(pattern_path.name):
        all_entries.extend(parse_image_list_file(file, include_intrinsics=with_intrinsics))
    return all_entries


def match_pair_name(name1: str, name2: str, sep: str = "/") -> str:
    return sep.join([name1.replace("/", "-"), name2.replace("/", "-")])


def match_pair_name_legacy(name1: str, name2: str) -> str:
    return match_pair_name(name1, name2, sep="_")


def locate_pair_key(hfile: h5py.File, name1: str, name2: str) -> Tuple[str, bool]:
    options = [
        (match_pair_name(name1, name2), False),
        (match_pair_name(name2, name1), True),
        (match_pair_name_legacy(name1, name2), False),
        (match_pair_name_legacy(name2, name1), True),
    ]
    for key, flipped in options:
        if key in hfile:
            return key, flipped
    raise KeyError(f"No valid pair key found for ({name1}, {name2})")


def load_matches(match_file: Path, name1: str, name2: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get matched keypoints and scores between two images from an HDF5 match file."""
    with h5py.File(str(match_file), "r") as f:
        key, flipped = locate_pair_key(f, name1, name2)
        raw_matches = f[key]["matches0"][()]
        raw_scores = f[key]["matching_scores0"][()]
    valid = raw_matches != -1
    src_idx = np.where(valid)[0]
    dst_idx = raw_matches[valid]
    matches = np.stack([src_idx, dst_idx], axis=-1)
    scores = raw_scores[valid]
    return (matches[:, ::-1], scores) if flipped else (matches, scores)
