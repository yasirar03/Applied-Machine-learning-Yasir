

import os
import json
import random
import shutil
import ssl
import tarfile
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    print("[WARN] certifi not installed - SSL downloads may fail on some machines. "
          "Run: pip install certifi")


# Global config

DATA_ROOT = os.environ.get("DATA_ROOT", "./data/datasets")



# AUTO-DOWNLOAD ENGINE

_SPLIT_JSON_GDRIVE_IDS = {
    "split_zhou_Caltech101.json":          "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN",
    "split_zhou_OxfordPets.json":          "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs",
    "split_zhou_OxfordFlowers.json":       "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT",
    "split_zhou_Food101.json":             "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl",
    "split_zhou_FGVCAircraft.json":        "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y",
    "split_zhou_StanfordCars.json":        "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT",
    "split_zhou_SUN397.json":              "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq",
    "split_zhou_DescribableTextures.json": "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
    "split_zhou_EuroSAT.json":             "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o",
    "split_zhou_UCF101.json":              None,
}


_DATASET_SPECS = {
    "caltech101": {
        "folder": "caltech-101",
        "archives": [
            (
                "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
                "zip",
                None,   # extracts as caltech-101/101_ObjectCategories/
            ),
        ],
        "split_json": "split_zhou_Caltech101.json",
        "check_path": "caltech-101/101_ObjectCategories",   # existence check
    },
    "oxfordpets": {
        "folder": "oxford_pets",
        "archives": [
            (
                "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                "tar.gz",
                ("images", "images"),
            ),
            (
                "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
                "tar.gz",
                ("annotations", "annotations"),
            ),
        ],
        "split_json": "split_zhou_OxfordPets.json",
        "check_path": "oxford_pets/images",
    },
    "flowers102": {
        "folder": "oxford_flowers",
        "archives": [
            (
                "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
                "tar.gz",
                ("jpg", "jpg"),
            ),
            (
                "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
                "none",
                None,
            ),
        ],
        "split_json": "split_zhou_OxfordFlowers.json",
        "check_path": "oxford_flowers/jpg",
    },
    "food101": {
        "folder": "food-101",
        "archives": [
            (
                "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "tar.gz",
                ("food-101", None),   # None = extract in-place, folder already named food-101
            ),
        ],
        "split_json": "split_zhou_Food101.json",
        "check_path": "food-101/images",
    },
    "fgvcaircraft": {
        "folder": "fgvc_aircraft",
        "archives": [
            (
                "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
                "tar.gz",
                ("fgvc-aircraft-2013b/data", "fgvc_aircraft_tmp"),
            ),
        ],
        "split_json": "split_zhou_FGVCAircraft.json",
        "check_path": "fgvc_aircraft/images",
    },
    "stanfordcars": {
        "folder": "stanford_cars",
        "archives": [
            (
                "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/data/cars_train.tgz",
                "tar.gz",
                ("cars_train", "cars_train"),
            ),
            (
                "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/data/cars_test.tgz",
                "tar.gz",
                ("cars_test", "cars_test"),
            ),
            (
                "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/data/car_devkit.tgz",
                "tar.gz",
                ("devkit", "devkit"),
            ),
        ],
        "split_json": "split_zhou_StanfordCars.json",
        "check_path": "stanford_cars/cars_train",
    },
    "sun397": {
        "folder": "sun397",
        "archives": [
            (
                "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
                "tar.gz",
                ("SUN397", "SUN397"),
            ),
        ],
        "split_json": "split_zhou_SUN397.json",
        "check_path": "sun397/SUN397",
    },
    "dtd": {
        "folder": "dtd",
        "archives": [
            (
                "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
                "tar.gz",
                ("dtd", None),   # extracts as dtd/ already
            ),
        ],
        "split_json": "split_zhou_DescribableTextures.json",
        "check_path": "dtd/images",
    },
    "eurosat": {
        "folder": "eurosat",
        "archives": [
            (
                "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
                "zip",
                ("EuroSAT", "2750"),   # rename EuroSAT -> 2750 (CoOp convention)
            ),
        ],
        "split_json": "split_zhou_EuroSAT.json",
        "check_path": "eurosat/2750",
    },
    "ucf101": {
        "folder": "ucf101",
        "archives": [
            (
                # Mid-frame images hosted on HuggingFace (public mirror of CoOp's Drive link)
                "https://huggingface.co/datasets/jxu124/coop-splits/resolve/main/UCF-101-midframes.zip",
                "zip",
                ("UCF-101-midframes", "UCF-101-midframes"),
            ),
        ],
        "split_json": "split_zhou_UCF101.json",
        "check_path": "ucf101/UCF-101-midframes",
    },
}

def _progress_hook(block_num, block_size, total_size):
    """Simple download progress bar printed to stdout."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_done  = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(f"\r    {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
    else:
        print(f"\r    {downloaded / 1_048_576:.1f} MB downloaded", end="", flush=True)


def _download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file from url to dest_path.
    Returns True on success, False on failure.
    Skips download if dest_path already exists.

    Uses a browser-like User-Agent header, since some servers (e.g.
    data.caltech.edu) return 403 Forbidden for urllib's default
    "Python-urllib/x.y" user agent, treating it as a bot request.
    """
    if dest_path.exists():
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  -> {dest_path}")
    try:
        request = urllib.request.Request(
            str(url),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 64
            downloaded = 0
            with open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    _progress_hook(downloaded // block_size + 1, block_size, total_size)
        print()   # newline after progress bar
        return True
    except urllib.error.URLError as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False
    except Exception as e:
        print(f"\n  [ERROR] Unexpected error: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def _extract_archive(archive_path: Path, extract_to: Path, method: str):
    """Extract a tar.gz, tar, or zip archive to extract_to directory."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting: {archive_path.name} -> {extract_to}")
    if method in ("tar.gz", "tar"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
    elif method == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    print(f"  Extracted.")


def _download_split_json(split_json_name: str, dest_dir: Path) -> bool:
    """
    Download the CoOp split JSON file from official Google Drive using gdown.
    gdown is already installed as part of Dassl's dependencies.
    """
    dest_path = dest_dir / split_json_name
    if dest_path.exists():
        return True

    file_id = _SPLIT_JSON_GDRIVE_IDS.get(split_json_name)
    if file_id is None:
        print(f"  [ERROR] No Google Drive ID known for: {split_json_name}")
        return False

    print(f"  Downloading split JSON via gdown: {split_json_name}")
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest_path), quiet=False)
        if dest_path.exists() and dest_path.stat().st_size > 100:
            print(f"  [OK] Downloaded: {split_json_name}")
            return True
        else:
            print(f"  [ERROR] gdown downloaded empty/missing file.")
            return False
    except Exception as e:
        print(f"  [ERROR] gdown failed: {e}")
        return False


def _is_dataset_ready(data_root: str, dataset_name: str) -> bool:
    """Check if a dataset is already downloaded and ready."""
    spec = _DATASET_SPECS[dataset_name]
    check = Path(data_root) / spec["check_path"]
    split = Path(data_root) / spec["folder"] / spec["split_json"]
    return check.exists() and split.exists()


def download_dataset(dataset_name: str, data_root: str = DATA_ROOT):
    """
    Download and set up a single dataset automatically.

    Steps:
      1. Check if already present -- skip if yes.
      2. Download each image archive from its official URL.
      3. Extract the archive into the correct folder structure.
      4. Download the CoOp split JSON from HuggingFace mirror.
      5. Clean up downloaded archive files.

    Args:
        dataset_name : one of the 10 supported dataset names
        data_root    : root directory to store all datasets
    """
    if dataset_name not in _DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(_DATASET_SPECS.keys())}")

    if _is_dataset_ready(data_root, dataset_name):
        print(f"  [{dataset_name}] Already downloaded. Skipping.")
        return

    spec       = _DATASET_SPECS[dataset_name]
    folder     = spec["folder"]
    dest_dir   = Path(data_root) / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir    = Path(data_root) / "_tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Downloading dataset: {dataset_name.upper()}")
    print(f"  Destination: {dest_dir}")
    print(f"{'='*60}")

    # Step 1: Download and extract each archive 
    for url, method, rename_rule in spec["archives"]:
        fname      = url.split("/")[-1].split("?")[0]
        archive_path = tmp_dir / fname

        # Download
        ok = _download_file(url, archive_path)
        if not ok:
            print(f"  [WARN] Skipping archive (download failed): {fname}")
            continue

        if method == "none":
            # Plain file -- just copy to dest_dir
            shutil.copy2(archive_path, dest_dir / fname)
            print(f"  Copied: {fname}")
        else:
            # Extract to tmp, then move to correct location
            extract_tmp = tmp_dir / f"extract_{fname}"
            _extract_archive(archive_path, extract_tmp, method)

            if rename_rule is not None:
                src_name, dst_name = rename_rule
                src_path = extract_tmp / src_name
                if src_path.exists():
                    if dst_name is None:
                        # Move contents directly into dest_dir
                        for item in src_path.iterdir():
                            target = dest_dir / item.name
                            if not target.exists():
                                shutil.move(str(item), str(target))
                    else:
                        dst_path = dest_dir / dst_name
                        if not dst_path.exists():
                            shutil.move(str(src_path), str(dst_path))
                        print(f"  Moved: {src_name} -> {dst_path}")
                else:
                    # Try moving everything from extract root
                    for item in extract_tmp.iterdir():
                        target = dest_dir / item.name
                        if not target.exists():
                            shutil.move(str(item), str(target))
            else:
                # No rename -- move everything from extract root to dest_dir
                for item in extract_tmp.iterdir():
                    target = dest_dir / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))

            # Clean up extract tmp
            shutil.rmtree(extract_tmp, ignore_errors=True)

        # Remove archive to save disk space
        archive_path.unlink(missing_ok=True)

    # Special post-processing
    if dataset_name == "fgvcaircraft":
        # fgvc-aircraft-2013b/data -> fgvc_aircraft/
        tmp_data = dest_dir / "fgvc_aircraft_tmp"
        if tmp_data.exists():
            for item in tmp_data.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(tmp_data, ignore_errors=True)

    if dataset_name == "food101":
        # food-101.tar.gz extracts as food-101/ inside dest_dir
        # Move contents up one level if nested
        nested = dest_dir / "food-101"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(nested, ignore_errors=True)

    if dataset_name == "dtd":
        # dtd.tar.gz extracts as dtd/ inside dest_dir -- move up
        nested = dest_dir / "dtd"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(nested, ignore_errors=True)

    if dataset_name == "caltech101":
        # caltech-101.zip extracts as caltech-101/ inside dest_dir -- move up
        nested = dest_dir / "caltech-101"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(nested, ignore_errors=True)
        # The zip also contains 101_ObjectCategories.tar.gz -- extract it
        inner_tar = dest_dir / "101_ObjectCategories.tar.gz"
        if inner_tar.exists() and not (dest_dir / "101_ObjectCategories").exists():
            print("  Extracting inner 101_ObjectCategories.tar.gz...")
            with tarfile.open(inner_tar, "r:gz") as tf:
                tf.extractall(dest_dir)
            inner_tar.unlink(missing_ok=True)
            print("  Extracted 101_ObjectCategories.")

    # Step 2: Download split JSON 
    print(f"  Downloading split JSON: {spec['split_json']}")
    ok = _download_split_json(spec["split_json"], dest_dir)
    if not ok:
        print(f"  [ERROR] Could not download split JSON for {dataset_name}.")
        print(f"          Please download manually from:")
        print(f"          https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md")

    # Cleanup tmp dir 
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Final check
    if _is_dataset_ready(data_root, dataset_name):
        print(f"\n  [OK] {dataset_name.upper()} ready at: {dest_dir}")
    else:
        print(f"\n  [FAIL] {dataset_name.upper()} setup may be incomplete.")
        print(f"    Expected: {Path(data_root) / spec['check_path']}")
        print(f"    Please check manually.")


def download_all_datasets(data_root: str = DATA_ROOT,
                           datasets: list = None):
    """
    Download ALL 10 datasets automatically.
    Already-downloaded datasets are skipped.

    Args:
        data_root : root directory (default: DATA_ROOT env var or ./data/datasets)
        datasets  : list of dataset names to download (default: all 10)

    Usage:
        python common/datasets.py                          # download all
        python common/datasets.py --datasets flowers102   # download one
    """
    if datasets is None:
        datasets = list(_DATASET_SPECS.keys())

    print(f"\nDataset root: {Path(data_root).resolve()}")
    print(f"Datasets to download: {datasets}")

    already_done = []
    to_download  = []
    for name in datasets:
        if _is_dataset_ready(data_root, name):
            already_done.append(name)
        else:
            to_download.append(name)

    if already_done:
        print(f"\nAlready present ({len(already_done)}): {already_done}")
    if not to_download:
        print("All datasets already downloaded.")
        return

    print(f"\nWill download ({len(to_download)}): {to_download}")
    total_size_note = {
        "caltech101":   "~125 MB",
        "oxfordpets":   "~800 MB",
        "flowers102":   "~330 MB",
        "food101":      "~4.7 GB",
        "fgvcaircraft": "~2.7 GB",
        "stanfordcars": "~1.8 GB",
        "sun397":       "~36 GB",
        "dtd":          "~600 MB",
        "eurosat":      "~90 MB",
        "ucf101":       "~6.5 GB",
    }
    total = sum(
        float(total_size_note[n].split()[0].replace("~", ""))
        for n in to_download
    )
    print(f"\nEstimated download sizes:")
    for n in to_download:
        print(f"  {n:<20} {total_size_note.get(n, 'unknown')}")
    print(f"  {'TOTAL':<20} ~{total:.0f} GB/MB (see above for units)")
    print()

    for name in to_download:
        try:
            download_dataset(name, data_root)
        except Exception as e:
            print(f"\n  [ERROR] Failed to download {name}: {e}")
            print(f"          Continuing with remaining datasets...")

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name in datasets:
        status = "[OK] Ready" if _is_dataset_ready(data_root, name) else "[FAIL] Incomplete"
        print(f"  {name:<20} {status}")


# END OF AUTO-DOWNLOAD ENGINE


# CLIP normalisation constants for ViT-B/16 (224x224)
# These are the official values from openai/CLIP -- no need to load the model here.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

TRAIN_TRANSFORM = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

TEST_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# Base Dataset

class BaseDataset(Dataset):
    """
    Generic dataset that reads (image_path, label) pairs from a list.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples   # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# Dataset Loaders -- each returns (train, val, test, classnames)


def _load_split_file(split_path):
    """Read CoOp-style split JSON: {train: [[path,label,classname],...], ...}"""
    with open(split_path, "r") as f:
        data = json.load(f)
    return data


def _build_classnames(items):
    """
    Build classnames list ordered by the label indices in the JSON.

    CRITICAL: The CoOp split JSONs use a fixed numeric label assignment
    that is NOT alphabetical order. For example in flowers102:
      label 0 = 'pink primrose', label 1 = 'hard-leaved pocket orchid', ...
    If we sort classnames alphabetically and use that as the index mapping,
    every prediction will be wrong because the indices don't match.

    This function builds classnames[label_idx] = classname correctly
    by reading the label->name mapping directly from the JSON items.
    """
    label_to_name = {}
    for item in items:
        label_to_name[int(item[1])] = item[2]
    max_label = max(label_to_name.keys())
    classnames = [label_to_name.get(i, f"class_{i}") for i in range(max_label + 1)]
    return classnames


def _build_samples(items, image_root=None):
    """
    Convert CoOp split items to (absolute_path, label_idx) tuples.

    The CoOp split JSONs store only relative filenames (e.g. 'image_00190.jpg'
    or 'apple_pie/3113710.jpg'). We prepend image_root so the DataLoader can
    open the files directly.

    Args:
        items      : list of [relative_path, label_int, classname] from JSON
        image_root : Path to prepend to each filename. If None, path is used as-is.
    """
    if image_root is None:
        return [(item[0], int(item[1])) for item in items]
    root = Path(image_root)
    return [(str(root / item[0]), int(item[1])) for item in items]


def load_caltech101(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "caltech-101"
    split = _load_split_file(root / "split_zhou_Caltech101.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "101_ObjectCategories"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_oxfordpets(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "oxford_pets"
    split = _load_split_file(root / "split_zhou_OxfordPets.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "images"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_flowers102(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "oxford_flowers"
    split = _load_split_file(root / "split_zhou_OxfordFlowers.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "jpg"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_food101(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "food-101"
    split = _load_split_file(root / "split_zhou_Food101.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "images"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_fgvcaircraft(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "fgvc_aircraft"
    split = _load_split_file(root / "split_zhou_FGVCAircraft.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "images"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_stanfordcars(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "stanford_cars"
    split = _load_split_file(root / "split_zhou_StanfordCars.json")
    classnames = _build_classnames(split["train"])
    train = _build_samples(split["train"], root)
    val   = _build_samples(split["val"],   root)
    test  = _build_samples(split["test"],  root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_sun397(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "sun397"
    split = _load_split_file(root / "split_zhou_SUN397.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "SUN397"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_dtd(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "dtd"
    split = _load_split_file(root / "split_zhou_DescribableTextures.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "images"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_eurosat(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "eurosat"
    split = _load_split_file(root / "split_zhou_EuroSAT.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "2750"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames


def load_ucf101(data_root=DATA_ROOT, shots=None):
    root = Path(data_root) / "ucf101"
    split = _load_split_file(root / "split_zhou_UCF101.json")
    classnames = _build_classnames(split["train"])
    image_root = root / "UCF-101-midframes"
    train = _build_samples(split["train"], image_root)
    val   = _build_samples(split["val"],   image_root)
    test  = _build_samples(split["test"],  image_root)
    if shots:
        train = _few_shot_sample(train, shots, len(classnames))
    return train, val, test, classnames



# Registry -- easy lookup by name

DATASET_LOADERS = {
    "caltech101":    load_caltech101,
    "oxfordpets":    load_oxfordpets,
    "flowers102":    load_flowers102,
    "food101":       load_food101,
    "fgvcaircraft":  load_fgvcaircraft,
    "stanfordcars":  load_stanfordcars,
    "sun397":        load_sun397,
    "dtd":           load_dtd,
    "eurosat":       load_eurosat,
    "ucf101":        load_ucf101,
}

ALL_DATASETS = list(DATASET_LOADERS.keys())


def get_dataset(name, data_root=DATA_ROOT, shots=None, auto_download=True):
    """
    Returns (train_samples, val_samples, test_samples, classnames).

    Args:
        name          : dataset name (see ALL_DATASETS)
        data_root     : root directory containing all dataset folders
        shots         : if int, returns few-shot training split (shots per class)
        auto_download : if True (default), automatically downloads the dataset
                        if it is not already present under data_root

    Auto-download behaviour:
        The first time you call get_dataset("flowers102"), it will download
        the images and CoOp split JSON automatically. Subsequent calls skip
        the download since the data is already present.
    """
    assert name in DATASET_LOADERS, \
        f"Unknown dataset: {name}. Choose from {ALL_DATASETS}"

    # Auto-download if not present
    if auto_download and not _is_dataset_ready(data_root, name):
        print(f"  Dataset '{name}' not found at {data_root}. Starting download...")
        download_dataset(name, data_root)

    
        if not _is_dataset_ready(data_root, name):
            spec = _DATASET_SPECS[name]
            raise RuntimeError(
                f"Dataset '{name}' is still not ready after attempting download. "
                f"Expected images at: {Path(data_root) / spec['check_path']}\n"
                f"Expected split JSON at: {Path(data_root) / spec['folder'] / spec['split_json']}\n"
                f"Check the download log above for the actual failure (SSL errors, "
                f"404s, timeouts, etc.) and either re-run once the underlying issue "
                f"is fixed, or download/extract '{name}' manually to that path."
            )

    return DATASET_LOADERS[name](data_root=data_root, shots=shots)


def get_dataloader(samples, transform, batch_size=64, shuffle=False, num_workers=None):
    """
    Build a DataLoader from a list of (path, label) samples.

    num_workers=None (default) auto-selects:
      - Windows: 0 -- multiprocessing workers are slow to spawn there due to
        process creation overhead (no fork(), spawn-based instead).
      - macOS: 0 -- also spawn-based (not fork), meaning every worker
        re-imports torch/torchvision/CLIP/common from scratch on startup.
        Combined with the MPS backend this has been observed to hang rather
        than just being slow, and for these dataset sizes the parallel
        workers rarely save more time than their startup overhead costs.
      - Linux: min(4, CPU count) -- fork-based, so workers start cheaply and
        parallel image decode/transform reliably helps.

    Pass num_workers explicitly to override this default in either
    direction (e.g. try num_workers=2 on macOS with CPU/CUDA - just avoid it
    with MPS).
    """
    import platform
    if num_workers is None:
        if platform.system() in ("Windows", "Darwin"):
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 1)
    dataset = BaseDataset(samples, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(num_workers > 0),
                      persistent_workers=(num_workers > 0))


def _few_shot_sample(samples, shots, n_classes, seed=42):
    """Sample `shots` examples per class from training data.

    Uses a dedicated Random instance seeded with `seed` (default 42) rather
    than the global `random` module, so:
      (a) the same few-shot split is reproducible across runs/methods -
          important when comparing baseline vs. weighted ensemble vs. CoOp
          on "the same" 16-shot training set, and
      (b) this doesn't perturb global random state used elsewhere (e.g. by
          other parts of the training pipeline that also call `random.*`).
    """
    from collections import defaultdict
    rng = random.Random(seed)
    class_samples = defaultdict(list)
    for path, label in samples:
        class_samples[label].append((path, label))
    result = []
    for label in range(n_classes):
        items = list(class_samples[label])
        rng.shuffle(items)
        result.extend(items[:shots])
    return result


# Run as script: python common/datasets.py [--datasets ...] [--data_root ...]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Auto-download all datasets for the EEEM068 project."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets to download. Default: all 10. Choices: {ALL_DATASETS}"
    )
    parser.add_argument(
        "--data_root", default=DATA_ROOT,
        help=f"Root directory to store datasets. Default: {DATA_ROOT}"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check which datasets are present -- do not download."
    )
    args = parser.parse_args()

    if args.check:
        print(f"\nDataset root: {Path(args.data_root).resolve()}")
        print(f"{'Dataset':<20} {'Status'}")
        print("-" * 35)
        for name in (args.datasets or ALL_DATASETS):
            status = "[OK] Ready" if _is_dataset_ready(args.data_root, name) \
                     else "[FAIL] Not found"
            print(f"  {name:<20} {status}")
    else:
        download_all_datasets(
            data_root=args.data_root,
            datasets=args.datasets
        )
