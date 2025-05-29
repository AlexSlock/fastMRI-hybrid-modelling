import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

import h5py
from tqdm import tqdm

import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query
import torch
import torch.nn.functional as F

# DL_MRI_reconstruction_baselines!!

# python run_zero_filled_cs.py \
#  --data_path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Preprocessed_CS/multicoil_test/ \
#  --output_path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/CS/ \

# ADDED (copied from preprocessed_transforms.py)
def downscale_bart_output(img_torch: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """
    Downscale a BART output image (real or complex) from 640x640 to target_size.
    Assumes the input is a 2D complex-valued TENSOR of shape or (H, W, 2) = output of T.to_tensor
    """
    img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)  # -> (1, 2, H, W)

    # Resize to 320x3taget_shape 20 using bilinear interpolation
    img_resized = F.interpolate(img_torch, target_size, mode='bilinear', align_corners=False)
    #img_resized = F.interpolate(img_torch, target_size, mode='area')

     # Convert back to (H, W, 2)
    img_resized = img_resized.squeeze(0).permute(1, 2, 0)  # -> (H, W, 2)

    return img_resized

def save_zero_filled(data_dir, out_dir):
    '''
    Takes CS reconstructions in preprocessed h5 files from data_dir
    - crops to size from encodedSpace?!
    - takes absolute value
    and saves as seperate h5 files with only classical CS reconstruction in out_dir
    '''
    reconstructions = {}

    ## CHANGED: select .npy files in data_dir
    for fname in tqdm(list(data_dir.glob("*.npy"))):
            cs_data = np.load(fname)
            ##### ADDED: find matching h5 file
            tgt_file = None
            target_paths = [Path("/DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test_full/"),
                                Path("/DATASERVER/MIC/SHARED/NYU_FastMRI/Knee/multicoil_val/")]
            for target_dir in target_paths:
                candidate = target_dir / fname.name.replace("_cs.npy", ".h5")
                if candidate.exists():
                    tgt_file = candidate
                    break
            assert tgt_file is not None, f"Target file not found for {fname.name}"
            #####

            with h5py.File(tgt_file, "r") as hf:
                et_root = etree.fromstring(hf["ismrmrd_header"][()])
                #image = transforms.to_tensor(hf["cs_data"][()])

                #### ADDED: reshape cs_data(for every slice)
                orig_shape = (hf["kspace"][0].shape[1], hf["kspace"][0].shape[2])
                # print(f"Original shape: {orig_shape}")
                # print(f"cs_data shape: {cs_data.shape}")

                # initialize zeros to hold images of shape (orig°shape[0], orig_shape[1], 2) for each slice of kspace
                image = torch.zeros((cs_data.shape[0], orig_shape[0], orig_shape[1], 2))
                # print(f"Initialized image shape: {image.shape}")

                for slice in range(cs_data.shape[0]):
                    cs_data_slice = transforms.to_tensor(cs_data[slice])
                    image[slice]= (downscale_bart_output(cs_data_slice, orig_shape))
                #####


                # extract target image width, height from ismrmrd header
                enc = ["encoding", "encodedSpace", "matrixSize"]        ## MISTAKE HERE??: should crop to "reconSpace", not encodedSpace
                crop_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                )

                # check for FLAIR 203
                if image.shape[-2] < crop_size[1]:
                    crop_size = (image.shape[-2], image.shape[-2])

                # crop input image
                image = transforms.complex_center_crop(image, crop_size)

                # absolute value
                image = fastmri.complex_abs(image)
                
                # change to tgt_file.name if you want .h5 files! (before was fname.name)
                reconstructions[tgt_file.name] = image # Dictionary mapping input filenames to corresponding reconstructions

    fastmri.save_reconstructions(reconstructions, out_dir) # comes from fastmri/utils.py
    # save_reconstructions(reconstructions:Dict[str, numpy.ndarray], out_dir:pathlib.Path)
    # => Save reconstruction images as h5 files.
    # This function writes to h5 files that are appropriate for submission to the
    # leaderboard.
    # Args:
    #     reconstructions: A dictionary mapping input filenames to corresponding
    #         reconstructions.
    #     out_dir: Path to the output directory where the reconstructions should
    #         be saved.


def create_arg_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    # NOT USED
    # parser.add_argument(
    #     "--challenge",
    #     type=str,
    #     required=True,
    #     help="Which challenge",
    # )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    # save_zero_filled(args.data_path, args.output_path, args.challenge)
    save_zero_filled(args.data_path, args.output_path)
    print("Finished Zero Filled reconstruction")
