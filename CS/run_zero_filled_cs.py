import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm

import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query


def save_zero_filled(data_dir, out_dir):
    '''
    Takes CS reconstructions in preprocessed h5 files from data_dir
    - crops to size from encodedSpace?!
    - takes absolute value
    and saves as seperate h5 files with only classical CS reconstruction in out_dir
    '''
    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            image = transforms.to_tensor(hf["cs_data"][()])

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

            reconstructions[fname.name] = image # Dictionary mapping input filenames to corresponding reconstructions

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
