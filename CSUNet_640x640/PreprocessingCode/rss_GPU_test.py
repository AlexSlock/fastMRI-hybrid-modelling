import os
import h5py
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
import time

from fastmri.fftc import ifft2c_new
from fastmri import rss_complex
from fastmri.data import transforms as T
from torch.cuda.amp import autocast


# Set the start method for multiprocessing
set_start_method('spawn', force=True)

# Set max_split_size_mb to limit fragmentation
torch.cuda.set_per_process_memory_fraction(0.8)  # Limit to 80% of GPU memory

def zero_pad_kspace(kspace, target_size=(640, 640)):
    is_3d = len(kspace.shape) == 3
    if not is_3d:
        kspace = kspace[np.newaxis, ...]

    rows, cols = kspace.shape[-2:]
    target_rows, target_cols = target_size

    kspace_tensor = T.to_tensor(kspace).to('cuda')

    if rows > target_rows and cols > target_cols:
        kspace_tensor = T.complex_center_crop(kspace_tensor, target_size)
        rows, cols = target_size
    if rows > target_rows:
        kspace_tensor = T.complex_center_crop(kspace_tensor, (target_rows, cols))
        rows = target_rows
    if cols > target_cols:
        kspace_tensor = T.complex_center_crop(kspace_tensor, (rows, target_cols))
        cols = target_cols

    if rows < target_rows or cols < target_cols:
        pad_rows = target_rows - rows
        pad_cols = target_cols - cols
        pad_top = pad_rows // 2
        pad_bottom = pad_rows - pad_top
        pad_left = pad_cols // 2
        pad_right = pad_cols - pad_left
        kspace_tensor = torch.nn.functional.pad(
            kspace_tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom)
        )

    kspace_padded = T.tensor_to_complex_np(kspace_tensor.cpu())
    return kspace_padded if is_3d else kspace_padded[0]

def compute_rss_target(kspace, target_size):
    kspace_padded = zero_pad_kspace(kspace, target_size=target_size)
    with autocast():  # Use mixed precision
        image_tensor = ifft2c_new(T.to_tensor(kspace_padded).to('cuda'))
        rss_image = rss_complex(image_tensor).cpu().numpy()
        
    torch.cuda.empty_cache()  # Clear GPU cache after each slice

    return rss_image

def process_file(file_path):
    print(f"Processing file: {file_path}")  # Debugging print
    basename = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_rss.pt")

    with h5py.File(file_path, 'r') as hf:
        kspace = hf['kspace'][:]
        orig_shape = kspace.shape[-2:]
        all_rss = []

        for slice_idx in range(kspace.shape[0]):
            rss = compute_rss_target(kspace[slice_idx], target_size)
            all_rss.append(rss)

        stacked = torch.tensor(np.stack(all_rss), dtype=torch.float32)
        torch.save({'image': stacked, 'orig_shape': orig_shape}, output_path)
        print(f"Saved RSS data to {output_path}")  # Debugging print


output_dir = "/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Full_RSS_target/test/"
input_dir = '/DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_train/'
num_workers = 20
target_size = (640, 640)
num_files = 20

os.makedirs(output_dir, exist_ok=True)
h5_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.h5')]

start = time.time()
with Pool(processes=min(num_workers, cpu_count())) as pool:
    list(tqdm(pool.imap(process_file, h5_files), total=num_files))

print('Total time for 20 files: ', time.time() - start)
