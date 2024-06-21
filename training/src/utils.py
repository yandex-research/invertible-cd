import os

import torch
from torchvision.transforms import ToPILImage


def recover_resume_step(output_dir):
    sample_dirs = [
        name for name in os.listdir(output_dir) if name.startswith("checkpoint")
    ]
    if len(sample_dirs) == 0:
        return 0
    last_samples = sorted(sample_dirs, key=lambda x: int(x.split("-")[1]))[-1]
    split = last_samples.split("-")
    if len(split) < 2:
        return 0
    try:
        return int(split[1])
    except ValueError:
        return 0


def find_resume_checkpoint(output_dir):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    path = os.path.join(output_dir, "model.pt")
    if os.path.exists(path):
        return path
    return None


def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    else:
        tensor = ((tensor + 1) / 2).clamp(0, 1)

    return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
