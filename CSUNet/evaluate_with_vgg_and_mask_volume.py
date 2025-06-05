import os
import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fastmri.data import transforms

import torch
from torchvision.models import vgg19
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Lambda
import json
from tqdm import tqdm
import json
import pandas as pd

# Run with conda DL_MRI_reconstruction_baselines_2

# python evaluate_with_vgg_and_mask_volume.py \
#   --target-paths /DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test_full/ \
#                 /DATASERVER/MIC/SHARED/NYU_FastMRI/Knee/multicoil_val/ \
#   --predictions-path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/CSUNet_brain/reconstructions/ \
#   --bart-path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Preprocessed_CS/multicoil_test/ \
#   --output-path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/evaluations/ \


## INITIALIZE GPU AND LOAD VGG19 MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load the pre-trained VGG19 model
vgg_model = vgg19(pretrained=True).features[:36].to(device).eval()


def determine_and_apply_mask(target, recons, tgt_file):
    """
    processes two reconstruction files and applies a mask to 
    the target and reconstructed images based on the intersection of 
    non-zero values of 2 != reconstructions (sense and CS).
    => goal: only evaluate where they have meaningful values 
    Args:
        target (np.ndarray): ground truth image
        recons (np.ndarray): reconstructed image
        tgt_file (pathlib.Path): path to the target file
    """
    # define the base paths for sense + CS reconstructions
    reconstruction_sense_path_string = '/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/Sense/'
    reconstruction_CS_path_string = '/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/CS/'
    # Construct full pahts by appending target file name
    reconstruction_sense_path = pathlib.Path(reconstruction_sense_path_string) / tgt_file.name
    reconstruction_CS_path = pathlib.Path(reconstruction_CS_path_string) / tgt_file.name
    # Read reconstruction files
    reconstruction_sense = h5py.File(reconstruction_sense_path, 'r')
    reconstruction_CS = h5py.File(reconstruction_CS_path, 'r')
    reconstruction_sense = reconstruction_sense['reconstruction']
    reconstruction_CS = reconstruction_CS['reconstruction']
    # Convert to numpy arrays
    reconstruction_sense = np.array(reconstruction_sense)
    reconstruction_CS = np.array(reconstruction_CS)
    # Crop the reconstructions to the same size as the target
    reconstruction_sense = transforms.center_crop(reconstruction_sense, (target.shape[-1], target.shape[-1]))
    reconstruction_CS = transforms.center_crop(reconstruction_CS, (target.shape[-1], target.shape[-1]))
    # Create bitmasks where non-zero values in the reconstructions are marked as 1, and zero values are marked as 0.
    sense_bitmask = np.ones_like(reconstruction_sense)
    sense_bitmask = np.where(reconstruction_sense != 0, sense_bitmask, 0).astype(int)
    CS_bitmask = np.ones_like(reconstruction_CS)
    CS_bitmask = np.where(reconstruction_CS != 0, CS_bitmask, 0).astype(int)
    # Create an intersection mask where the non-zero values in the sense and CS reconstructions overlap
    intersection_mask = CS_bitmask & sense_bitmask
    # Apply the intersection mask to the target and reconstructed images
    gt = np.where(intersection_mask == 1, target, 0)
    pred = np.where(intersection_mask == 1, recons, 0)
        # If the value in intersection_mask is 1, the corresponding value from target is retained.
        # If the value in intersection_mask is 0, the corresponding value in gt is set to 0.
    return gt, pred


# Define the preprocessing steps for the VGG loss
preprocess = Compose([
    ToTensor(),
    CenterCrop((224, 224)), # Ensure the center part of the image is used
    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def vgg_loss(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = gt * 255
    pred = pred * 255

    losses = []

    for gt_img, pred_img in zip(gt, pred):
        gt_tensor = preprocess(gt_img).unsqueeze(0).to(device)
        pred_tensor = preprocess(pred_img).unsqueeze(0).to(device)

        gt_feat = vgg_model(gt_tensor)
        pred_feat = vgg_model(pred_tensor)

        loss = torch.nn.functional.mse_loss(gt_feat, pred_feat)
        losses.append(loss)
    
    return torch.stack(losses).mean().cpu().numpy()

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    VGG=vgg_loss,
    #SVD=stacked_svd,
)


class Metrics:
    def __init__(self, metric_funcs):
        self.metric_funcs = metric_funcs
        self.metrics = {metric: Statistics() for metric in metric_funcs}
        self.per_volume = []
    
    def push(self, target, recons, filename=None, acquisition=None, R=None):
        metric_result = {}
        for metric, func in self.metric_funcs.items():
            val = func(target, recons)
            self.metrics[metric].push(val)
            metric_result[metric] = val
        self.per_volume.append({
            "filename": filename,
            "acquisition": acquisition,
            "R": R,
            **metric_result,
        })

    def get_per_volume(self):
        return self.per_volume

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

        
    # load acceleartion factors once
    acc_factor_file = pathlib.Path(args.bart_path) / "acceleration_factors.json"
    with open(acc_factor_file, "r") as f:
        acc_factors = json.load(f)
    
    # TODO: change for prostate as well when needed
    # changed for loop => iterate over files in predictions_path and look for target files in both knee + brain directory!
    # changed iterdir() to glob() to only look for .h5 files (cause of .json file)
    for pred_file in tqdm(sorted(args.predictions_path.glob("*.h5")), desc="Evaluating"):
        ###  find matching target file (knee or brain)
        tgt_file = None
        for target_dir in args.target_paths:
            candidate = target_dir / pred_file.name
            if candidate.exists():
                tgt_file = candidate
                break
        assert tgt_file is not None, f"Target file not found for {pred_file.name}"
        ###

        with h5py.File(tgt_file, "r") as target, h5py.File(pred_file, "r") as recons:

             # used if only test 1 type of acquisition
            target_acquisition = target.attrs["acquisition"]
            if args.acquisition and args.acquisition != target_acquisition:
                continue
            
            # always compute R
            R = float(acc_factors.get(pred_file.name, -1)) # stored in json file for knee data
            if R < 0:
                # Handle brain case by loading mask
                mask_path = tgt_file.parent.parent / "multicoil_test" / tgt_file.name
                with h5py.File(mask_path, 'r') as mask:
                    nPE_mask = mask['mask'][()]
                sampled_columns = np.sum(nPE_mask)
                R = len(nPE_mask) / sampled_columns

            # (filter after) used if only test 1 type of acceleration
            if args.acceleration:
                # ignore file if R is not within +-0.1 of the target acceleration factor (too small margin?!)
                # changed => see fastmri/test_data_accelerations.ipynb => see that max margin is 0.26
                target_R = float(args.acceleration)
                if abs(R - target_R) > 0.26:
                    continue
            
            # select target and reconstruction
            target = target[recons_key][()] # "reconstruction_rss" of target files exists in multicoil_test_full set!
            recons = recons["reconstruction"][()]

            # center crop the images to the size of the target
            target = transforms.center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
            recons = transforms.center_crop(
                recons, (target.shape[-1], target.shape[-1])
            )
            # apply non-zero mask to target and reconstruction
            target, recons = determine_and_apply_mask(target, recons, tgt_file)
            # calculate metrics
            metrics.push(
                target, recons,
                filename=pred_file.stem,
                acquisition=target_acquisition,
                R=round(R)
            )

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-paths",
        type=pathlib.Path,
        nargs="+",  # Accept one or more paths
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--bart-path",
        type=pathlib.Path,
        required=True,
        help="Path to preprocessed BART data (for acceleration factors)",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=None,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        default="multicoil",
        help="Which challenge",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=None,
        help= "If set, only volumes with specified acceleration factor are used. If not set, all "
        "acceleration factors are evaluated. ",
    )
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args = parser.parse_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    metrics = evaluate(args, recons_key)
    print(metrics)


    if args.output_path is not None:
        # Ensure output directory exists
        args.output_path.mkdir(parents=True, exist_ok=True)
        model_name = args.predictions_path.parts[-1]  # assuming the model name is the parent directory of predictions_path
        if model_name == "reconstructions":
            model_name = args.predictions_path.parts[-2]  # if it's just "reconstructions", take the parent directory
        volume_csv_out = args.output_path / f"metrics_per_volume_{model_name}.csv"
        df = pd.DataFrame(metrics.get_per_volume())  # assuming it's a list of dicts
        df.to_csv(volume_csv_out, index=False)
        print(f"Saved per-volume metrics to: {volume_csv_out}")

        

