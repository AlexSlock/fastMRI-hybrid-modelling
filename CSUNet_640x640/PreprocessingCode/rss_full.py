import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from fastmri.fftc import ifft2c_new
from fastmri import rss_complex
from fastmri.data import transforms as T
import yaml


def zero_pad_kspace(kspace, target_size=(640, 640)):
    """
    Zero-pad k-space to achieve sinc interpolation in image domain
    
    Args:
        kspace (numpy.array): K-space data (can be 2D or 3D with num_coils)
        target_size (tuple): Target size (rows, cols)
    
    Returns:
        kspace_padded (numpy.array): Zero-padded k-space
    """
    is_3d = len(kspace.shape) == 3  # Check if num_coils dimension exists
    if not is_3d:
        kspace = kspace[np.newaxis, ...]  # Add dummy coil dimension

    rows, cols = kspace.shape[-2], kspace.shape[-1]
    target_rows, target_cols = target_size

    # Convert k-space to fastMRI expected format (real, imag) -> shape (num_coils, rows, cols, 2)
    kspace_tensor = T.to_tensor(kspace)

     # Handle cropping if the size is too large
    if rows > target_rows and cols > target_cols:
        kspace_tensor = T.complex_center_crop(kspace_tensor, target_size)
        rows = target_rows
        cols = target_cols

     # if only 1 dimension is too large, crop that dimension   
    if rows > target_rows:
        kspace_tensor = T.complex_center_crop(kspace_tensor, (target_rows, cols))
        rows = target_rows
    
    if cols > target_cols:
        kspace_tensor = T.complex_center_crop(kspace_tensor, (rows, target_cols))
        cols = target_cols

    # Handle zero-padding if the size is too small
    if rows < target_rows or cols < target_cols:
        pad_rows = target_rows - rows
        pad_cols = target_cols - cols
        
        pad_top = pad_rows // 2
        pad_bottom = pad_rows - pad_top
        pad_left = pad_cols // 2
        pad_right = pad_cols - pad_left
        
        # Apply zero padding to 3D array
        kspace_tensor = torch.nn.functional.pad(
            kspace_tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom)
        )

    kspace_padded = T.tensor_to_complex_np(kspace_tensor)
    return kspace_padded if is_3d else kspace_padded[0]  # Remove dummy coil dimension if needed

def compute_full_rss_target(kspace, target_size=(640, 640)):
    """
    Reconstructs a fully-sampled image from zero-padded k-space and computes RSS.
    Args:
        kspace (ndarray): Fully-sampled k-space (shape: [coils, height, width])
        target_size (tuple): Desired output image size (e.g. (640, 640))
    Returns:
        dict: Dictionary containing:
            - 'image': Zero-padded RSS image of numpy array shape [H, W]
            - 'orig_shape': Original shape of the k-space data
    
    """
    kspace = zero_pad_kspace(kspace, target_size)
    image = rss_complex(ifft2c_new(T.to_tensor(kspace)))

    return image.numpy()

def process_volume(h5_path, save_dir, target_size=(640, 640)):
    start_time_file = time.time()
    filename = os.path.basename(h5_path).replace('.h5', '')
    with h5py.File(h5_path, 'r') as hf:
        kspace = hf['kspace']
        orig_shape = kspace.shape[-2:]
        num_slices = kspace.shape[0]
        all_rss = []

        for slice_idx in range(num_slices):
            kspace_slice = kspace[slice_idx]
            rss = compute_full_rss_target(kspace_slice, target_size)
            all_rss.append(rss)

        stacked = torch.tensor(np.stack(all_rss), dtype=torch.float32)
        # Save to disk
        save_path = os.path.join(save_dir, f'{filename}_rss.pt')
        torch.save({'image': stacked, 'orig_shape': orig_shape}, save_path)

        end_time_file = time.time()
        print(f"Time taken to process {filename}: {end_time_file - start_time_file:.2f} seconds")

# Process all files in a dataset folder
def process_dataset_parallel(data_dir, save_dir, target_size=(640, 640), max_workers=8):
    os.makedirs(save_dir, exist_ok=True)
    h5_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')])

    process_fn = partial(process_volume, save_dir=save_dir, target_size=target_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, h5_path): h5_path for h5_path in h5_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel RSS Gen"):
            result = future.result()
            #print(result)  # only printed 'None'

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load configuration
    # assumes the config file is named rss_full_config.yaml 
    # and is in the same directory as your script.
    config = load_config("rss_full_config.yml") 
    print(config)

    # Use values from config
    data_dir = config["data_dir"]
    save_dir = config["save_dir"]
    target_size = config["target_size"]
    workers = config["workers"]

    start = time.time()
    process_dataset_parallel(data_dir, save_dir, target_size, max_workers=workers)
    print('Total time for all files: ', time.time() - start)


if __name__ == "__main__":
    main()