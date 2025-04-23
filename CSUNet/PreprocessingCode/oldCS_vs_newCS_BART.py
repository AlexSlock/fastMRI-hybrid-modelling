import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from fastmri.data import transforms as T
import fastmri
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

train_path = '/DATASERVER/MIC/SHARED/NYU_FastMRI/Preprocessed/multicoil_train/'
CS_path = '/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/Preprocessed_CS/multicoil_train/'

num_threads = 25

def downscale_bart_output(img: np.ndarray, target_size) -> np.ndarray:
    img_torch = torch.tensor(img)
    if torch.is_complex(img_torch):
        img_torch = torch.view_as_real(img_torch).permute(2, 0, 1).unsqueeze(0)
    else:
        img_torch = img_torch.unsqueeze(0).unsqueeze(0)
    img_resized = F.interpolate(img_torch, target_size, mode='bilinear', align_corners=False)
    img_resized = img_resized.squeeze(0)
    if img_resized.shape[0] == 2:
        return torch.view_as_complex(img_resized.permute(1, 2, 0).contiguous()).numpy()
    else:
        return img_resized.squeeze(0).numpy()

def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    return ssim(image1, image2, data_range=image1.max())

def process_slice(fname: str, slice_idx: int):
    try:
        with h5py.File(os.path.join(train_path, fname), 'r') as f:

            original_CS = f['cs_data'][()]
            if slice_idx >= original_CS.shape[0]:
                return None

            npy_path = os.path.join(CS_path, fname.replace('.h5','') + '_cs.npy')
            if not os.path.exists(npy_path):
                print(f"File not found: {npy_path}")
                return None
            new_CS = np.load(npy_path)
            if slice_idx >= new_CS.shape[0]:
                return None
            ## target data:
            target = f['reconstruction_rss'][slice_idx]

            resized_new_CS = downscale_bart_output(new_CS[slice_idx], (original_CS.shape[1], original_CS.shape[2]))

            prep_orig_CS = T.to_tensor(original_CS[slice_idx])
            prep_new_CS = T.to_tensor(resized_new_CS)
            

            if target is not None:
                crop_size = (target.shape[-2], target.shape[-1])
                #print('crop_size:', crop_size)
            else:
                crop_size = (f.attrs["recon_size"][0], f.attrs["recon_size"][1])
                #print('crop_size:', crop_size)
            # check for FLAIR 203
            if prep_new_CS.shape[-2] < crop_size[1]:
                crop_size = (prep_new_CS.shape[-2], prep_new_CS.shape[-2])
                #print('NEW crop_size:', crop_size)

            prep_orig_CS = T.complex_center_crop(prep_orig_CS, crop_size)
            prep_new_CS = T.complex_center_crop(prep_new_CS, crop_size)

            prep_orig_CS = fastmri.complex_abs(prep_orig_CS)
            prep_new_CS = fastmri.complex_abs(prep_new_CS)

            prep_orig_CS, _, _ = T.normalize_instance(prep_orig_CS, eps=1e-11)
            prep_orig_CS = prep_orig_CS.clamp(-6, 6)
            prep_new_CS, _, _ = T.normalize_instance(prep_new_CS, eps=1e-11)
            prep_new_CS = prep_new_CS.clamp(-6, 6)

            ssim_val = compute_ssim(prep_orig_CS.numpy(), prep_new_CS.numpy())
            return (fname, slice_idx, ssim_val)
    except Exception as e:
        return None

# Prepare slice jobs
tasks = []
for fname in os.listdir(CS_path):
    fname = fname.replace('_cs.npy','.h5') 
    if not fname.endswith('.h5'):
        continue
    if 'brain' not in fname:
        continue ## ONLY SELECT BRAIN FILES
    try:
        with h5py.File(os.path.join(train_path, fname), 'r') as f:
            original_CS = f['cs_data'][()]
            tasks.extend([(fname, i) for i in range(original_CS.shape[0])])
    except Exception as e:
        print(f"Error reading {fname}: {e}")

# Run parallel processing
results = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(process_slice, fname, idx): (fname, idx) for fname, idx in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slices"):
        res = future.result()
        if res:
            results.append(res)

# Group and average by volume
ssim_per_volume = defaultdict(list)
for fname, slice_idx, ssim_val in results:
    ssim_per_volume[fname].append(ssim_val)

volume_avg = {fname: np.mean(ssims) for fname, ssims in ssim_per_volume.items()}
overall_avg = np.mean(list(volume_avg.values()))

# Print per-volume and final average
print("=== Average SSIM per Volume ===")
for fname, avg in sorted(volume_avg.items(), key=lambda x: x[1], reverse=True):
    print(f"{fname}: {avg:.4f}")

print(f"\n✅ Final Overall Average SSIM: {overall_avg:.4f}")

df = pd.DataFrame(list(volume_avg.items()), columns=["filename", "avg_ssim"])
df.to_csv("volume_avg_ssim.csv", index=False)

print("Saved to volume_avg_ssim.csv ✅")
