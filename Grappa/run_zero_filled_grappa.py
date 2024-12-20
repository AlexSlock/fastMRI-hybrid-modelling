import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm

import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query

# python run_zero_filled_grappa.py --data_path /DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/ --output_path /DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Results/Reconstructions/Grappa/ --challenge “multicoil”

def save_zero_filled(data_dir, out_dir, which_challenge):
    '''
    Takes GRAPPA reconstructions in preprocessed h5 files from data_dir
    - does IFFT to get zero filled solution
    - crops to size from encodedSpace?!
    - takes absolute value
    - applies Root-Sum-of-Squares if multicoil data
    and saves as seperate h5 files with only classical GRAPPA reconstruction in out_dir
    '''
    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            masked_kspace = transforms.to_tensor(hf["grappa_data"][()])

            # extract target image width, height from ismrmrd header
            enc = ["encoding", "encodedSpace", "matrixSize"]
            crop_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
            )

            # inverse Fourier Transform to get zero filled solution
            image = fastmri.ifft2c(masked_kspace)

            # check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            # crop input image
            image = transforms.complex_center_crop(image, crop_size)

            # absolute value
            image = fastmri.complex_abs(image)

            # apply Root-Sum-of-Squares if multicoil data
            if which_challenge == "multicoil":
                image = fastmri.rss(image, dim=1)

            reconstructions[fname.name] = image

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
    parser.add_argument(
        "--challenge",
        type=str,
        required=True,
        help="Which challenge",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.output_path, args.challenge)
