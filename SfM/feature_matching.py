import h5py
import torch
from tqdm import tqdm
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional, Union
from threading import Thread
from queue import Queue
import collections.abc as collections

from .utils import parse_image_list_file, read_pairs_from_file, match_pair_name, match_pair_name_legacy, extract_image_names_from_h5, logger
from .deep_front_end import LightGlue

def generate_pairs(
    output: Path,
    image_list: Optional[Union[Path, List[str]]] = None,
    features: Optional[Path] = None,
    ref_list: Optional[Union[Path, List[str]]] = None,
    ref_features: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_list_file(Path(image_list))
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = extract_image_names_from_h5(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    self_matching = False
    if ref_list is not None:
        if isinstance(ref_list, (str, Path)):
            names_ref = parse_image_list_file(Path(ref_list))
        elif isinstance(ref_list, collections.Iterable):
            names_ref = list(ref_list)
        else:
            raise ValueError(f"Unknown type for reference image list: {ref_list}")
    elif ref_features is not None:
        names_ref = extract_image_names_from_h5(ref_features)
    else:
        self_matching = True
        names_ref = names_q

    pairs = []
    for i, n1 in enumerate(names_q):
        for j, n2 in enumerate(names_ref):
            if self_matching and j <= i:
                continue
            pairs.append((n1, n2))

    logger.info(f"Generated {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(f"{i} {j}" for i, j in pairs))

    return pairs


class ThreadedWriter:
    def __init__(self, fn, num_workers=4):
        self.fn = fn
        self.queue = Queue()
        self.threads = [Thread(target=self.worker) for _ in range(num_workers)]
        for t in self.threads:
            t.start()

    def put(self, item):
        self.queue.put(item)

    def join(self):
        for _ in self.threads:
            self.queue.put(None)
        for t in self.threads:
            t.join()

    def worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            self.fn(item)


class FeatureMatchDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], fpath_q: Path, fpath_r: Path):
        self.pairs = pairs
        self.fpath_q = fpath_q
        self.fpath_r = fpath_r

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        with h5py.File(self.fpath_q, "r") as f0, h5py.File(self.fpath_r, "r") as f1:
            data = {}
            for k, v in f0[name0].items():
                data[f"{k}0"] = torch.from_numpy(v[...]).float()
            for k, v in f1[name1].items():
                data[f"{k}1"] = torch.from_numpy(v[...]).float()
            data["image0"] = torch.empty((1,) + tuple(f0[name0]["image_size"])[::-1])
            data["image1"] = torch.empty((1,) + tuple(f1[name1]["image_size"])[::-1])
        return data


def write_match_result(item, out_path: Path):
    pair_name, prediction = item
    with h5py.File(out_path, "a", libver="latest") as f:
        if pair_name in f:
            del f[pair_name]
        group = f.create_group(pair_name)
        group.create_dataset("matches0", data=prediction["matches0"][0].cpu().short().numpy())
        if "matching_scores0" in prediction:
            group.create_dataset("matching_scores0", data=prediction["matching_scores0"][0].cpu().half().numpy())


def deduplicate_pairs(pairs: List[Tuple[str, str]], match_path: Optional[Path]) -> List[Tuple[str, str]]:
    seen = set()
    deduped = []
    for a, b in pairs:
        if (b, a) not in seen:
            seen.add((a, b))
            deduped.append((a, b))

    if match_path and match_path.exists():
        with h5py.File(match_path, "r", libver="latest") as f:
            final = []
            for a, b in deduped:
                if any(
                    match_pair_name(x, y) in f or match_pair_name_legacy(x, y) in f
                    for x, y in [(a, b), (b, a)]
                ):
                    continue
                final.append((a, b))
            return final

    return deduped

@torch.no_grad()
def run_lightglue_matching(
    image_pairs: Path,
    output_path: Path,
    feature_q: Path,
    feature_r: Optional[Path] = None,
    allow_overwrite: bool = False,
) -> None:
    logger.info("Running matching using LightGlue with DISK features.")

    if feature_r is None:
        feature_r = feature_q

    if not image_pairs.exists():
        raise FileNotFoundError(f"Missing image pair file: {image_pairs}")
    if not feature_q.exists() or not feature_r.exists():
        raise FileNotFoundError("Feature file(s) not found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    retrieval_dict = read_pairs_from_file(image_pairs)
    pair_list = [(q, r) for q, rs in retrieval_dict.items() for r in rs]
    pair_list = deduplicate_pairs(pair_list, None if allow_overwrite else output_path)

    if not pair_list:
        logger.info("All pairs already matched. Nothing to do.")
        return

    dataset = FeatureMatchDataset(pair_list, feature_q, feature_r)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = LightGlue({"name": "lightglue", "features": "disk"}).eval().to(device)

    writer = ThreadedWriter(partial(write_match_result, out_path=output_path), num_workers=4)

    for idx, batch in enumerate(tqdm(loader, smoothing=0.1)):
        batch = {k: (v if k.startswith("image") else v.to(device, non_blocking=True)) for k, v in batch.items()}
        prediction = matcher(batch)
        pair_name = match_pair_name(*pair_list[idx])
        writer.put((pair_name, prediction))

    writer.join()
    logger.info(f"Finished writing matches to {output_path}")

def generate_and_match_pairs(
    pair_file: Path,
    feature_q: Path,
    match_output: Path,
    image_list: Optional[Union[Path, List[str]]] = None,
    ref_list: Optional[Union[Path, List[str]]] = None,
    feature_r: Optional[Path] = None,
    allow_overwrite: bool = False,
):
    generate_pairs(
        output=pair_file,
        image_list=image_list,
        features=feature_q,
        ref_list=ref_list,
        ref_features=feature_r,
    )

    run_lightglue_matching(
        image_pairs=pair_file,
        output_path=match_output,
        feature_q=feature_q,
        feature_r=feature_r,
        allow_overwrite=allow_overwrite,
    )
