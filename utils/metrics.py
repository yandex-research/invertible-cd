import torch
import math
import numpy as np
import ImageReward as RM
import os
import pathlib
import shutil
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from transformers import AutoProcessor, AutoModel, AutoImageProcessor
from utils.inception import InceptionV3
from utils.generation import load_512, to_pil_images
from utils.loading import load_benchmark
from tqdm import tqdm
import json
import piq

# FID
# ------------------------------------------------------------
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, transforms=None):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def get_activations(images, model, batch_size=50, dims=2048, device='cpu', num_workers=8):
    model.eval()
    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    dataset = ImageDataset(images, #transforms=TF.ToTensor())
                            transforms=TF.Compose([
                                TF.Resize(256, interpolation=TF.InterpolationMode.LANCZOS), 
                                TF.CenterCrop(256), 
                                TF.ToTensor()]
                            ))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(images, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=8):
    act = get_activations(images, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=8):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s

def save_statistics_of_path(path, out_path, device=None, batch_size=50, dims=2048, num_workers=8):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = compute_statistics_of_path(path, model, batch_size, dims, device, num_workers)
    np.savez(out_path, mu=m1, sigma=s1)
    
def calculate_fid(
    images,
    path, 
    device=None, 
    batch_size=40,
    dims=2048, 
    num_workers=4, 
    inception_path="files/pt_inception-2015-12-05-6726825d.pth"
):
    """Calculates the FID of two paths"""
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], inception_path=inception_path).to(device)

    m1, s1 = calculate_activation_statistics(images, model, batch_size,
                                             dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(path, model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
# ------------------------------------------------------------


@torch.no_grad()
def calc_dinov2_images_images(images_1,
                              images_2,
                              device,
                              batch_size=50):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').eval().to(device)
    image_inputs_1 = processor(
        images=images_1,
        return_tensors="pt",
    )['pixel_values'].to(device)
    image_inputs_2 = processor(
        images=images_2,
        return_tensors="pt",
    )['pixel_values'].to(device)
    
    assert len(image_inputs_1) == len(image_inputs_2)
    
    scores = torch.zeros(len(image_inputs_2))
    for i in range(0, len(image_inputs_2), batch_size):
        image_batch_1 = image_inputs_1[i:i+batch_size]
        image_batch_2 = image_inputs_2[i:i+batch_size]
        # embed
        with torch.cuda.amp.autocast():
            image_embs_1 = model(pixel_values=image_batch_1).pooler_output
        image_embs_1 = image_embs_1 / torch.norm(image_embs_1, dim=-1, keepdim=True)
    
        with torch.cuda.amp.autocast():
            image_embs_2 = model(pixel_values=image_batch_2).pooler_output
        image_embs_2 = image_embs_2 / torch.norm(image_embs_2, dim=-1, keepdim=True)
        # score
        scores[i:i+batch_size] = (image_embs_2 * image_embs_1).sum(-1)
    return scores.cpu()


@torch.no_grad()
def calc_clip_score_images_images(images_1,
                                  images_2,
                                  device,
                                  batch_size=50):
    processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
    clip_model = AutoModel.from_pretrained('openai/clip-vit-large-patch14').eval().to(device)
    image_inputs_1 = processor(
        images=images_1,
        return_tensors="pt",
    )['pixel_values'].to(device)
    image_inputs_2 = processor(
        images=images_2,
        return_tensors="pt",
    )['pixel_values'].to(device)
    
    assert len(image_inputs_1) == len(image_inputs_2)

    scores = torch.zeros(len(image_inputs_2))
    for i in range(0, len(image_inputs_2), batch_size):
        image_batch_1 = image_inputs_1[i:i+batch_size]
        image_batch_2 = image_inputs_2[i:i+batch_size]
        # embed
        with torch.cuda.amp.autocast():
            image_embs_1 = clip_model.get_image_features(image_batch_1)
        image_embs_1 = image_embs_1 / torch.norm(image_embs_1, dim=-1, keepdim=True)
    
        with torch.cuda.amp.autocast():
            image_embs_2 = clip_model.get_image_features(image_batch_2)
        image_embs_2 = image_embs_2 / torch.norm(image_embs_2, dim=-1, keepdim=True)
        # score
        scores[i:i+batch_size] = (image_embs_2 * image_embs_1).sum(-1) #model.logit_scale.exp() * 
    return scores.cpu()


@torch.no_grad()
def calc_clip_score_images_prompts(images,
                                   prompts,
                                   device,
                                   batch_size=50):
    processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
    clip_model = AutoModel.from_pretrained('openai/clip-vit-large-patch14').eval().to(device)
    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )['pixel_values'].to(device)
    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )['input_ids'].to(device)
    
    assert len(image_inputs) == len(text_inputs)

    scores = torch.zeros(len(text_inputs))
    for i in range(0, len(text_inputs), batch_size):
        image_batch = image_inputs[i:i+batch_size]
        text_batch = text_inputs[i:i+batch_size]
        # embed
        with torch.cuda.amp.autocast():
            image_embs = clip_model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        with torch.cuda.amp.autocast():
            text_embs = clip_model.get_text_features(text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i:i+batch_size] = (text_embs * image_embs).sum(-1) #model.logit_scale.exp() * 
    return scores.cpu()

@torch.no_grad()
def calc_ir(images,
            prompts,
            device,
            batch_size=50):
    imagereward_model = RM.load("ImageReward-v1.0").eval().to(device)

    image_reward = []
    for prompt, image in zip(prompts, images):
        image_reward.append(imagereward_model.score(prompt, [image]))
    
    return image_reward

def calculate_psnr(images_1,
                   images_2,
                   device,
                   batch_size=50):
    psnr = []
    for img1, img2 in zip(images_1, images_2):
        img1 = np.array(img1).astype(np.float64)
        img2 = np.array(img2).astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        psnr_local = 20 * math.log10(255.0 / math.sqrt(mse))
        psnr.append(psnr_local)
    return psnr

def calculate_lpips(images_1,
                   images_2,
                   device,
                   batch_size=50):
    
    scores = torch.zeros(len(images_1))
    for i in tqdm(range(0, len(images_1), batch_size)):
        image_batch_1 = np.concatenate([np.array(i.resize((224,224))).reshape(1, 224, 224, 3) for i in images_1[i:i+batch_size]])
        image_batch_1 = torch.tensor(image_batch_1).permute(0, 3, 1, 2) / 255.
        image_batch_2 = np.concatenate([np.array(i.resize((224,224))).reshape(1, 224, 224, 3) for i in images_2[i:i+batch_size]])
        image_batch_2 = torch.tensor(image_batch_2).permute(0, 3, 1, 2) / 255.
        lpips_loss = piq.LPIPS(reduction='none')(image_batch_1.to(device), 
                                                 image_batch_2.to(device))
        scores[i:i+batch_size] = lpips_loss
    return scores


def calc_all(path_to_orig,
             path_to_edited,
             path_to_benchmark, 
             outdir,
             device):
    editing_benchmark = load_benchmark(path_to_benchmark,
                                       path_to_orig)
    files = os.listdir(path_to_edited)
    if len(files) == 0:
        print(path_to_edited)
        shutil.rmtree(path_to_edited)
        return None
    
    files = [file for file in files if file != 'editing_metrics_values.json']
    files.sort(key = lambda x: int(x.split('.')[0]))
    
    eval_collection = {'orig_prompt': [], 'orig_image': [], 'edited_prompt': [], 'edited_image': []}
    for j, (image_path, prompts_dict, _) in enumerate(tqdm(editing_benchmark)):
        #print(prompts_dict, f'{image_path}', f'{path_to_edited}/{files[j]}')
        #print('=======')
        try:
            orig_img = to_pil_images(load_512(f'{image_path}'))
            edited_img = to_pil_images(load_512(f'{path_to_edited}/{files[j]}'))
            eval_collection['orig_prompt'].append(prompts_dict['before'])
            eval_collection['orig_image'].append(orig_img)
            eval_collection['edited_prompt'].append(prompts_dict['after'])
            eval_collection['edited_image'].append(edited_img)
        except IndexError:
            continue
        
    try:
        preserve_clip_score = calc_clip_score_images_images(eval_collection['orig_image'], 
                                                                    eval_collection['edited_image'], 
                                                                    device='cuda', batch_size=16)
        print(f'cs {torch.mean(preserve_clip_score)}')
        print(f'cs {torch.std(preserve_clip_score)}')
        preserve_dinov2 = calc_dinov2_images_images(eval_collection['orig_image'], 
                                                            eval_collection['edited_image'], 
                                                            device='cuda', batch_size=16)
        print(f'dinov {torch.mean(preserve_dinov2)}')
        print(f'dinov {torch.std(preserve_dinov2)}')
        editing_clip_score = calc_clip_score_images_prompts(eval_collection['edited_image'], 
                                                                    eval_collection['edited_prompt'], 
                                                                    device='cuda', batch_size=16)
        print(f'cs edit {torch.mean(editing_clip_score)}')
        print(f'cs edit {torch.std(editing_clip_score)}')
        
        editing_imagereward = calc_ir(eval_collection['edited_image'], 
                                              eval_collection['edited_prompt'], 
                                              device='cuda', batch_size=16)
        print(f'ir {np.mean(editing_imagereward)}')
        print(f'ir {np.std(editing_imagereward)}')
        
        results = {'preservation_clip_score': str(list(np.array(preserve_clip_score))), 
                   'preservation_dinov2': str(list(np.array(preserve_dinov2))),
                   'editing_clip_score': str(list(np.array(editing_clip_score))),
                   'editing_imagereward': str(editing_imagereward)}

        os.makedirs(outdir, exist_ok=True)
        with open(f'{outdir}/editing_metrics_values.json', "w") as fp:
            json.dump(results , fp)
        
    except AssertionError:
        pass
    
def calc_inversion(path_to_dir,
                   device):
    
    files_1 = os.listdir(f'{path_to_dir}/generated_images')
    files_2 = os.listdir(f'{path_to_dir}/real_images')
    
    eval_collection = {'orig_image': [], 'edited_image': []}
    for j in tqdm(range(len(files_2))):
        #print(prompts_dict, f'{image_path}', f'{path_to_edited}/{files[j]}')
        #print('=======')
        try:
            orig_img = to_pil_images(load_512(f'{path_to_dir}/generated_images/{files_1[j]}'))
            edited_img = to_pil_images(load_512(f'{path_to_dir}/real_images/{files_2[j]}'))
            eval_collection['orig_image'].append(orig_img)
            eval_collection['edited_image'].append(edited_img)
        except IndexError:
            continue
        
    try:
        preserve_dinov2 = calc_dinov2_images_images(eval_collection['orig_image'], 
                                                            eval_collection['edited_image'], 
                                                            device='cuda', batch_size=16)
        print(torch.mean(preserve_dinov2))
        preserve_psnr = calculate_psnr(eval_collection['orig_image'], 
                                       eval_collection['edited_image'], 
                                       device='cuda', batch_size=16)
        print(np.mean(preserve_psnr))
        preserve_lpips = calculate_lpips(eval_collection['orig_image'], 
                                       eval_collection['edited_image'], 
                                       device='cuda', batch_size=16)
        print(torch.mean(preserve_lpips))
        
        results = {
                   'preservation_dinov2': str(list(np.array(preserve_dinov2))),
                   'preservation_psnr': str(list(np.array(preserve_psnr))),
                   'preservation_lpips': str(list(np.array(preserve_lpips))),
        }

        os.makedirs(path_to_dir, exist_ok=True)
        with open(f'{path_to_dir}/preservation_metrics_values.json', "w") as fp:
            json.dump(results , fp)
        
    except AssertionError:
        pass

