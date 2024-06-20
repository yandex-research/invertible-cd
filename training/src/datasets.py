import csv
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


def get_coco_loader(args, batch_size=50, is_train=False, max_cnt=None):
    """Load either train or val dataloader for the COCO2014 dataset"""
    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1,
        ]
    )
    dataset = COCODataset(
        args.coco_path,
        subset_name="train2014" if is_train else "val2014",
        transform=transform,
        max_cnt=max_cnt,
    )
    assert (
        max_cnt is None or len(dataset) == max_cnt
    ), f"COCO dataset size is {len(dataset)}/{max_cnt}"

    sampler_class = InfiniteSampler if is_train else DistributedSampler
    dataset_sampler = sampler_class(
        dataset=dataset,
        rank=dist.get_rank(),
        shuffle=is_train,
        num_replicas=dist.get_world_size(),
    )
    data = iter(torch.utils.data.DataLoader(
        dataset=dataset, sampler=dataset_sampler, batch_size=batch_size
    ))
    return data


class COCODataset(Dataset):
    def __init__(self, root_dir, subset_name="subset", transform=None, max_cnt=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )
        sample_dir = os.path.join(root_dir, subset_name)

        # Collect sample paths
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname[-4:] in self.extensions
            ],
            key=lambda x: x.split("/")[-1].split(".")[0],
        )
        self.samples = (
            self.samples if max_cnt is None else self.samples[:max_cnt]
        )  # restrict num samples

        # Collect captions
        self.captions = {}
        with open(
            os.path.join(root_dir, f"{subset_name}.csv"), newline="\n"
        ) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        return {
            "image": sample,
            "text": self.captions[os.path.basename(sample_path)],
            "idxs": idx,
        }


class InfiniteSampler(torch.utils.data.Sampler):
    """
    Distributed sampler iterating over the dataset forever
    """

    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
