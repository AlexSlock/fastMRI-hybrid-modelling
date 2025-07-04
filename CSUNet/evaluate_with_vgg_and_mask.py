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

# Run with conda DL_MRI_reconstruction_baselines

# python evaluate_with_vgg_and_mask.py \
#  --target-paths /DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test_full/ \
#   /DATASERVER/MIC/SHARED/NYU_FastMRI/Knee/multicoil_val/ \
#  --predictions-path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/CSUNet_brain/reconstructions/ \
#   --bart-path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Preprocessed_CS/multicoil_test/ \
#  --challenge multicoil \

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

def vgg_loss(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute VGG loss metric."""
    # Load the pre-trained VGG19 model
    vgg = vgg19(pretrained=True).features

    # Remove the last max pooling layer to get the feature maps
    vgg = torch.nn.Sequential(*list(vgg.children())[:-1])

    # Initialize a list to store the losses for each image in the batch
    losses = []

    # Convert inputs to the expected pixel range for RGB networks
    gt = gt*255
    pred = pred*255

    # Loop over each image in the batch
    for gt_image, pred_image in zip(gt, pred):
        # Preprocess the images
        gt_image = preprocess(gt_image)
        pred_image = preprocess(pred_image)

        # Ensure the images are batched
        gt_image = gt_image.unsqueeze(0)
        pred_image = pred_image.unsqueeze(0)

        # Extract features
        gt_features = vgg(gt_image)
        pred_features = vgg(pred_image)

        # Calculate VGG loss for the current pair of images
        loss = torch.nn.functional.mse_loss(gt_features, pred_features)
        losses.append(loss)

    # Average the losses across all images in the batch
    avg_loss = torch.mean(torch.stack(losses))

    return avg_loss.detach().cpu().numpy()


def stacked_svd(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute the average number of Singular Values required 
    to explain 90% of the variance in the residual error maps 
    of the reconstruction
    """
    residual_error_map = (gt-pred)**2
    U, S, Vh = np.linalg.svd(residual_error_map, full_matrices=True)
    num_slices = S.shape[0]
    im_size = S.shape[-1]
    singular_values_1d = S.flatten()
    abs_core = np.abs(singular_values_1d)
    sorted_indices = abs_core.argsort()[::-1]
    sorted_core = abs_core[sorted_indices]

    total_variance = np.sum(np.abs(sorted_core))

    # Calculate the cumulative sum of singular values
    cumulative_sum = np.cumsum(np.abs(sorted_core))

    num_svs = np.where(cumulative_sum >= 0.9*total_variance)[0][0] + 1

    num_svs_average = num_svs / num_slices

    return num_svs_average / im_size


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
    SVD=stacked_svd,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

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

    
    # TODO: change for loop => iterate over files in recon_path and look for target files in both knee + brain directory!
    for pred_file in args.predictions_path.iterdir():
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
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue
            
            # used if only test 1 type of acceleration
            if args.acceleration:
                # for knee data, AF is stored in json file
                R = float(acc_factors.get(pred_file.name, -1))
                if R < 0:
                    # for brain data, mask is stored in tgt_file (via the mask)
                    # filename = tgt_file.name
                    # mask_path = '/DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/'
                    # mask = h5py.File(os.path.join(mask_path,filename),'r')
                    # nPE_mask = mask['mask'][()]
                    nPE_mask = target['mask'][()]
                    sampled_columns = np.sum(nPE_mask)
                    R = len(nPE_mask)/sampled_columns
                    R = float(R)

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
            metrics.push(target, recons)

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
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
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
