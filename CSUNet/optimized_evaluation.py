import os
import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Lambda
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from fastmri.data import transforms
from concurrent.futures import ThreadPoolExecutor


# python optimized_evaluation.py --target-path /DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test_full/ --predictions-path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/Reconstructions/CSUNet/reconstructions/ --gpu-id 1 --acceleration 4.0

def set_default_gpu():
    # Set the default GPU to GPU # 
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"  # Force GPU number
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")


# Define preprocessing steps for VGG loss
preprocess = Compose([
    CenterCrop((224, 224)),
    Lambda(lambda x: x if x.size(1)!=1 else x.repeat(1, 3, 1, 1)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MRIDataset(Dataset):
    def __init__(self, target_path, predictions_path, recons_key, crop_size):
        self.target_files = list(pathlib.Path(target_path).iterdir())
        self.predictions_path = pathlib.Path(predictions_path)
        self.recons_key = recons_key
        self.crop_size = crop_size

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, idx):
        tgt_file = self.target_files[idx]
        with h5py.File(tgt_file, "r") as target, h5py.File(
            self.predictions_path / tgt_file.name, "r"
        ) as recons:
            target_data = target[self.recons_key][()]
            recons_data = recons["reconstruction"][()]
            target_data = transforms.center_crop(target_data, self.crop_size)
            recons_data = transforms.center_crop(recons_data, self.crop_size)
        return target_data, recons_data, tgt_file.name

def determine_and_apply_mask(target, recons, tgt_file_name):
    reconstruction_sense_path_string = '/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/Reconstructions/Sense/'
    reconstruction_CS_path_string = '/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/Reconstructions/CS/'
    reconstruction_sense_path = pathlib.Path(reconstruction_sense_path_string) / tgt_file_name
    reconstruction_CS_path = pathlib.Path(reconstruction_CS_path_string) / tgt_file_name
 

    with h5py.File(reconstruction_sense_path, 'r') as sense_file, h5py.File(reconstruction_CS_path, 'r') as cs_file:
        reconstruction_sense = torch.tensor(sense_file['reconstruction'][:], device='cuda')
        reconstruction_CS = torch.tensor(cs_file['reconstruction'][:], device='cuda')

     # Move tensors to CPU for transforms
    reconstruction_sense_cpu = reconstruction_sense.cpu()
    reconstruction_CS_cpu = reconstruction_CS.cpu()

    # Crop the reconstructions to the same size as the target using transforms.center_crop
    crop_size = target.shape[-1]  # Assuming square images
    reconstruction_sense_cpu = transforms.center_crop(reconstruction_sense_cpu, (crop_size, crop_size))
    reconstruction_CS_cpu = transforms.center_crop(reconstruction_CS_cpu, (crop_size, crop_size))

    # Move tensors back to GPU
    reconstruction_sense = reconstruction_sense_cpu.to('cuda')
    reconstruction_CS = reconstruction_CS_cpu.to('cuda')

    sense_bitmask = torch.tensor(reconstruction_sense != 0, dtype=torch.bool, device='cuda')
    CS_bitmask = torch.tensor(reconstruction_CS != 0, dtype=torch.bool, device='cuda')
    intersection_mask = sense_bitmask & CS_bitmask

    gt = torch.where(intersection_mask, target, torch.tensor(0.0, device='cuda'))
    pred = torch.where(intersection_mask, recons, torch.tensor(0.0, device='cuda'))
    return gt, pred

def vgg_loss(gt: torch.Tensor, pred: torch.Tensor, device: torch.device) -> torch.Tensor:
    vgg = vgg19(pretrained=True).features.to(device).eval()
    gt = preprocess(gt * 255).unsqueeze(0).to(device)
    pred = preprocess(pred * 255).unsqueeze(0).to(device)
    gt_features = vgg(gt)
    pred_features = vgg(pred)
    return torch.nn.functional.mse_loss(gt_features, pred_features)

def compute_metrics_concurrently(target, recons, metrics_funcs, device):
    results = {}

    def compute_metric(name, func):
        if name == "VGG":
            return name, func(target, recons, device).item()
        else:
            return name, func(target.cpu().numpy(), recons.cpu().numpy())

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_metric, name, func) for name, func in metrics_funcs.items()]
        for future in futures:
            name, value = future.result()
            results[name] = value

    return results

METRIC_FUNCS = {
    "MSE": lambda gt, pred: np.mean((gt - pred) ** 2),
    "NMSE": lambda gt, pred: np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2,
    "PSNR": lambda gt, pred: peak_signal_noise_ratio(gt, pred, data_range=gt.max()),
    "SSIM": lambda gt, pred: np.mean([
        structural_similarity(gt[i], pred[i], data_range=gt.max()) for i in range(gt.shape[0])
    ]),
    "VGG": vgg_loss,
}

def evaluate(args):
    set_default_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available
    dataset = MRIDataset(args.target_path, args.predictions_path, args.recons_key, crop_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    metrics_results = {name: [] for name in METRIC_FUNCS}

    for target, recons, tgt_file_names in dataloader: 
        target, recons = target.to(device), recons.to(device) 
        
        # Process each file name if tgt_file_names is a list 
        for tgt_file_name in tgt_file_names: 
            if args.acceleration: 
                mask_path = pathlib.Path('/DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/') / tgt_file_name 
                with h5py.File(mask_path, 'r') as mask_file: 
                    nPE_mask = mask_file['mask'][:] 
                    sampled_columns = np.sum(nPE_mask) 
                    R = len(nPE_mask) / sampled_columns 
                    if not (args.acceleration - 0.1 <= R <= args.acceleration + 0.1): 
                        continue            

        # Apply mask using Sense and CS reconstructions
        target, recons = determine_and_apply_mask(target, recons, tgt_file_name)

        # Compute metrics
        metrics = compute_metrics_concurrently(target, recons, METRIC_FUNCS, device)
        for name, value in metrics.items():
            metrics_results[name].append(value)

    # Average metrics over all samples
    metrics_summary = {name: np.mean(values) for name, values in metrics_results.items()}
    return metrics_summary

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target-path", type=pathlib.Path, required=True, help="Path to the ground truth data")
    parser.add_argument("--predictions-path", type=pathlib.Path, required=True, help="Path to reconstructions")
    parser.add_argument("--recons-key", type=str, default="reconstruction_rss", help="Key for reconstruction data")
    parser.add_argument("--acceleration", type=float, default=None, help="Target acceleration factor for filtering")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify which GPU to use")
    args = parser.parse_args()

    metrics = evaluate(args)
    print(metrics)
