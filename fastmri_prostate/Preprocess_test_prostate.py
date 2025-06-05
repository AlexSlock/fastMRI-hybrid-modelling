import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFunc
import time
import gc
import bart
import os
import logging

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import json
import scipy.io as sio

# nohup Preprocess_test_prostate.py > output2.log 2>&1 &
def zero_pad_calibration(calibration_data, target_shape):
    """Zero-pads the calibration data to match full k-space shape."""
    S, C, RO_calib, PE_calib = calibration_data.shape
    RO_full, PE_full = target_shape

    padded = np.zeros((S, C, RO_full, PE_full), dtype=calibration_data.dtype)
    start_ro = (RO_full - RO_calib) // 2
    start_pe = (PE_full - PE_calib) // 2

    padded[:, :, start_ro:start_ro + RO_calib, start_pe:start_pe + PE_calib] = calibration_data
    return padded

def flip_im(vol, slice_axis):
    """
    Flips a 3D image volume along the slice axis.

    Parameters
    ----------
    vol : numpy.ndarray of shape (slices, height, width)
        The 3D image volume to be flipped.
    slice_axis : int
        The slice axis along which to perform the flip

    Returns
    -------
    numpy.ndarray
        The flipped 3D image volume 
    """

    for i in range(vol.shape[slice_axis]):
        vol[i] = np.flipud(vol[i])
    return vol

def process_wrapper(args):
    fname, acceleration, save_dir, mat_file = args
    return process_volume(fname, save_dir, mat_file, acceleration)


def apply_mask(slice_kspace, mask_func):
    ''' 
    Args:
        slice_kspace (numpy.array)
        mask_func (class)
    Returns:
        masked_kspace (numpy.array)
        mask (torch.tensor)
    '''
    slice_kspace_T = T.to_tensor(slice_kspace)
    masked_kspace_T, mask = T.apply_mask(slice_kspace_T, mask_func, seed=42) # Use seed for validation data
    masked_kspace = T.tensor_to_complex_np(masked_kspace_T)
    return masked_kspace, mask

def generate_array(shape, R, mat_file, tensor_out):
    '''
    Generate CS mask for given k_space shape and acceleration factor R
    Args:
        shape (tuple): Shape of the k-space data
        R (int): Acceleration factor
        mat_file: matlab file containing the CS masks
        tensor_out (bool): If True, the output will be a torch tensor
    Returns:
        array (numpy.array or torch.tensor): CS mask 
    '''
    if R == 4:
        array = mat_file['m320_CS4_mask'].squeeze()
    elif R == 8:
        array = mat_file['m320_CS8_mask'].squeeze()
    else:
        raise ValueError('Unrecognized acceleration factor specified. Must be 4 or 8.')
    # Calculate padding needed to reach the desired length
    desired_length = shape[-1]
    padding_needed = desired_length - len(array)
    if padding_needed > 0:
        # Calculate padding width for symmetric padding
        padding_width = (padding_needed // 2, padding_needed - padding_needed // 2)
        # Pad the array symmetrically
        array = np.pad(array, padding_width, mode='symmetric')
    elif padding_needed < 0:
        # Calculate trimming indices for symmetric trimming
        trim_start = -padding_needed // 2
        trim_end = len(array) + padding_needed // 2
        # Trim the array symmetrically
        array = array[trim_start:trim_end]
    # Make array compatible with fastmri mask function class
    for i in range(len(shape)-1):
        array = np.expand_dims(array, 0)
    if tensor_out:
        array = T.to_tensor(array)
    return array


def estimate_sensitivity_maps(kspace):
    ''' 
    Args:
        kspace (numpy.array): slice kspace of shape (num_coils, rows, cols)
    Returns:
        S (numpy.array): Estimated sensitivity maps given by ESPIRiT of shape (num_coils, rows, cols)
    '''
    # Move coil axis to the back as expected by BART
    kspace_perm = np.moveaxis(kspace, 0, 2)
    # Add extra dimension, because the ESPIRiT method expects a 4D input array where the third dimension represents the batch size.
    kspace_perm = np.expand_dims(kspace_perm, axis=2)
    # Estimate sensitivity maps with ESPIRiT method
    S = bart.bart(1, "ecalib -d0 -m1", kspace_perm)
    # Undo the previous operations to get the original data structure back
    S = np.moveaxis(S.squeeze(), 2, 0)
    return S

def CS(kspace, S, lamda=0.005, num_iter=50):
    ''' 
    Performs CS reconstruction
    https://mrirecon.github.io/bart/

    Args:
        kspace (numpy.array): Slice kspace of shape (num_coils, rows, cols)
        S (numpy.array): Estimated sensitivity maps given by ESPIRiT of shape (num_coils, rows, cols)
        lamda: Value of the hyperparameter / regularizer of the l1 norm term
        num_iter: The amount of iterations the algorithm can run
    Returns:
        reconstruction (numpy.array): Estimated CS reconstruction of shape (rows, cols))
    '''
    # Move coil axis to the back as expected by BART
    kspace_perm = np.moveaxis(kspace, 0, 2)
    S_perm = np.moveaxis(S, 0, 2)
    # Add extra dimension, because BART expects a 4D input array where the third dimension represents the batch size.
    kspace_perm = np.expand_dims(kspace_perm, axis=2)
    S_perm = np.expand_dims(S_perm, axis=2)
    # Perform CS
    reconstruction = bart.bart(1, 'pics -S -l1 -r {} -i {} -d 0'.format(lamda, num_iter), kspace_perm, S_perm)
    return reconstruction

def process_volume(fname, save_dir, mat_file, acceleration):
    # Timer for the entire file
    start_time_file = time.time()   

    # Open HDF5 file in read mode
    with h5py.File(fname, 'r') as hf:
        kspace = hf['kspace'][:]
        calibration_data = hf['calibration_data'][:]

    num_averages, num_slices, num_coils, RO, PE = kspace.shape
    cs_recons = np.zeros((num_averages, num_slices, RO, PE), dtype=np.complex64)
    
    # pad calibration data to match full k-space shape
    calib_padded = zero_pad_calibration(calibration_data, target_shape=(RO, PE))

    # chose undersampling pattern !!for ACS region in calibration data!! based on acceleration factor
    if acceleration == 4:
        mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4])
    else:
        mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[8])

    # === estimate sensitivity maps ONCE for all averages from calibration data ===
    S_all = []
    masked_calib, _ = apply_mask(calib_padded, mask_func) # make sure seed is set (unless training)
    for slice in range(num_slices):
        S = estimate_sensitivity_maps(masked_calib[slice,:,:,:]) # (num_coils, RO_calib, PE_calib)
        S_all.append(S)


    # === PROSTATE: 3 averages, calculate CS for each before averaging to get result
    for avg in range(num_averages):
        avg_kspace = kspace[avg] # (num_slices, num_coils, nx, ny)
        
        # Apply CS mask
        mask = generate_array(avg_kspace.shape, acceleration, mat_file, tensor_out=False)
        masked_kspace = avg_kspace * mask + 0.0

        # compute CS reconstruction per slice
        for slice in range(num_slices):
            cs_recons[avg, slice] = CS(masked_kspace[slice], S_all[slice])

    # === average across the 3 averages ===
    cs_data = np.mean(cs_recons, axis=0)  # Average across the first dimension (averages)

    # === flip vertically to match ground truth orientation ===
    cs_data = flip_im(cs_data, slice_axis=0) # flip each slice individually

    # Save the CS data to an HDF5 file
    output_file = Path(save_dir) / fname.name
    with h5py.File(output_file, 'w') as hf_out:
        hf_out.create_dataset('reconstruction', data=cs_data)

    time.sleep(1)
    del kspace, masked_kspace, mask, cs_data, cs_recons, calib_padded, S_all, calibration_data, masked_calib
    gc.collect()
    time.sleep(1)

    logging.info(f"  Saved CS data to {output_file}")

    # Timer for the entire file: calculate and logging.info total elapsed time after all slices
    end_time_file = time.time()
    elapsed_time_file = end_time_file - start_time_file
    logging.info(f"Total time for processing the entire file: {elapsed_time_file:.4f} seconds")

    return (fname.name, acceleration)



def process_dataset_parallel(data_dir, save_dir, mat_file, max_workers):
    os.makedirs(save_dir, exist_ok=True)

    ##################### TEST PROSTATE #######################
    # Select first 139 T2 files
    t2_folders = [f for f in os.listdir(data_dir) if 'T2' in f]
    h5_files = []
    for folder in t2_folders:
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            h5_files.extend(list(Path(folder_path).glob("*.h5")))
    
    h5_files = h5_files[:139]  # Limit to first 139 files for testing

    logging.info(f"Number of files to process: {len(h5_files)}")

    ### ADD to save AF + ensure (accurate!) fifty fifty division
    n_files = len(h5_files)

    # Create a balanced acceleration factor list: half 4s, half 8s
    half = n_files // 2
    af_list = [4] * half + [8] * (n_files - half)

    np.random.seed(42)  # Set seed for reproducibility => get SENSE the same AF for same files!
    np.random.shuffle(af_list)  # Shuffle once to randomize order

    # We'll assign AF from af_list by index matching files
    # Pair files with assigned acceleration factor
    files_with_af = list(zip(h5_files, af_list))
    args_list = [(fname, acc, save_dir, mat_file) for fname, acc in files_with_af]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_wrapper, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel CS Gen"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing file {futures[future][0]}: {e}")
    
    # Save acceleration factors mapping to JSON
    json_file_path = os.path.join(save_dir, "acceleration_factors.json")
    acc_dict = {fname: acc for fname, acc in results}
    with open(json_file_path, 'w') as jf:
        json.dump(acc_dict, jf, indent=4)
    logging.info(f"Saved acceleration factors for {len(results)} files to {json_file_path}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    logging.basicConfig(filename='preprocessing_prostate2.log', level=logging.INFO)
    logging.info('Started processing')
    # Load configuration
    # assumes the config file is named rss_full_config.yaml 
    # and is in the same directory as your script.
    config = load_config("preprocess_prostate_config.yml") 

    # Use values from config
    data_dir = config["data_dir"]
    save_dir = config["save_dir"]
    mat_dir = config["mat_dir"]
    mat_file = sio.loadmat(mat_dir)
    workers = config["workers"]

    start = time.time()
    process_dataset_parallel(data_dir, save_dir, mat_file, max_workers=workers)
    logging.info('Total time for all files: {:.2f} seconds'.format(time.time() - start))
    logging.info('Finished processing')

if __name__ == "__main__":
    main()

