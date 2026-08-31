"""Datasets: STL-10 (simple study) and ImageNet-100 (scaled study).

Augmentation policy:
- masked objectives (mae/genmask/ijepa*): RandomResizedCrop + hflip only,
  as in I-JEPA ("without hand-crafted view augmentations").
- jea: SimCLR-strength two-view augs (crop, jitter, grayscale, blur-lite),
  so its collapse cannot be blamed on weak augmentations.
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

STL_MEAN, STL_STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TwoView:
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        return self.t(x), self.t(x)


def ssl_transform(arch, img_size, mean, std):
    if arch == "jea":
        aug = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return TwoView(aug)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def eval_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224))
        if img_size >= 128 else transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class HFImageDataset(Dataset):
    """Wraps a HuggingFace image dataset split (e.g. clane9/imagenet-100)."""

    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        return self.transform(img), ex["label"]


def get_ssl_dataset(name, arch, root):
    if name == "stl10":
        t = ssl_transform(arch, 96, STL_MEAN, STL_STD)
        return datasets.STL10(root, split="unlabeled", download=True, transform=t)
    if name == "in100":
        from datasets import load_dataset
        t = ssl_transform(arch, 224, IN_MEAN, IN_STD)
        ds = load_dataset("clane9/imagenet-100", split="train",
                          cache_dir=os.path.join(root, "hf"))

        class _Wrap(Dataset):
            def __len__(self):
                return len(ds)

            def __getitem__(self, i):
                img = ds[i]["image"].convert("RGB")
                return t(img), 0
        return _Wrap()
    raise ValueError(name)


def get_probe_datasets(name, root):
    """Returns (train_ds, test_ds) yielding (img, label) with eval transform."""
    if name == "stl10":
        t = eval_transform(96, STL_MEAN, STL_STD)
        return (datasets.STL10(root, split="train", download=True, transform=t),
                datasets.STL10(root, split="test", download=True, transform=t))
    if name == "in100":
        from datasets import load_dataset
        t = eval_transform(224, IN_MEAN, IN_STD)
        cache = os.path.join(root, "hf")
        tr = load_dataset("clane9/imagenet-100", split="train", cache_dir=cache)
        va = load_dataset("clane9/imagenet-100", split="validation",
                          cache_dir=cache)
        return HFImageDataset(tr, t), HFImageDataset(va, t)
    raise ValueError(name)
