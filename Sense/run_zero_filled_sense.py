import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm

import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query

# ADDED SO CAN RUN ON GPU:
import torch
import os

# python run_zero_filled_sense.py --data_path /DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/ --output_path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/Reconstructions/Sense/

def set_default_gpu():
    # Set the default GPU to GPU #1 if CUDA_VISIBLE_DEVICES is not set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force GPU #1
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

def save_zero_filled(data_dir, out_dir):
    '''
    Takes SENSE reconstructions in preprocessed h5 files from data_dir
    - crops to size from encodedSpace?!
    - takes absolute value
    - and saves as seperate h5 files with only classical SENSE reconstruction in out_dir
    '''

    set_default_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available

    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            image = transforms.to_tensor(hf["sense_data"][()]).to(device)   # ADDED .to(device) => move to GPU

            # extract target image width, height from ismrmrd header
            enc = ["encoding", "encodedSpace", "matrixSize"]
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

            reconstructions[fname.name] = image.cpu()  # ADDED .cpu() (move back to cpu before saving)

    fastmri.save_reconstructions(reconstructions, out_dir)


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
    #save_zero_filled(args.data_path, args.output_path, args.challenge)
    save_zero_filled(args.data_path, args.output_path)
