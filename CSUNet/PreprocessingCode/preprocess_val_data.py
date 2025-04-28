import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFunc
import torch
import time
import gc
import bart
import scipy.io as sio
import random
import os
import logging

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

def fifty_fifty():
    return random.random() < .5

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
    # why now use a seed in validation and not in training? 
    # needed for fair comparison between != training models
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

def process_volume(fname, save_dir, mat_file):
    # Timer for the entire file
    start_time_file = time.time()   

    # Open HDF5 file in read mode
    with h5py.File(fname, 'r') as hf:
        kspace = hf['kspace'][()]

    #print("Shape of the raw kspace: ", str(np.shape(kspace)))

    # Define target resolution for zero-padding/cropping
    target_size = (640, 640)  # Modify if needed

    # Randomly decide if R = 4 or 8 for equispaced mask => ACS region for estimating coil sensitivities! 
    undersampling_bool = fifty_fifty()  
    if undersampling_bool:
        mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4])
    else:
        mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[8])
    masked_kspace_ACS, mask_ACS = apply_mask(kspace, mask_func)
    #print("Shape of the generated ACS mask: ", str(mask_ACS.shape))

    # same random if R = 4 or 8 for CS mask
    if undersampling_bool:
        mask = generate_array(kspace.shape, 4, mat_file, tensor_out=False)
    else:
        mask = generate_array(kspace.shape, 8, mat_file, tensor_out=False)
    # (following = OK, see Transforms.apply_mask)
    masked_kspace = kspace * mask + 0.0   # +0.0 removes the sign of the zeros
    #print("Shape of the generated CS mask: ", str(mask.shape))


    ##################### CHANGED: SO WORKS WITH zero-padding/cropping BEFORE BART ############
    # Perform CS reconstruction
    cs_data = np.zeros((kspace.shape[0], target_size[0], target_size[1]), dtype=np.complex64)
    for slice in range(kspace.shape[0]):
        # timer for each slice
        start_time_slice = time.time()

        # Zero-fill/crop before ESPIRiT sensitivity estimation
        padded_kspace_ACS = zero_pad_kspace(masked_kspace_ACS[slice, :, :, :], target_size)
        padded_kspace = zero_pad_kspace(masked_kspace[slice, :, :, :], target_size)
        #print("Shape of the padded kspace: ", str(np.shape(padded_kspace)))

         # Estimate sensitivity maps
        S_padded = estimate_sensitivity_maps(padded_kspace_ACS) # estimate Si with ACS region

        # Perform CS reconstruction with zero-filled k-space
        cs_data[slice, :, :] = CS(padded_kspace, S_padded)
        end_time_slice = time.time()
        elapsed_time_slice = end_time_slice - start_time_slice
        #print(f"Time for slice: {elapsed_time_slice:.4f} seconds")
    #print("Shape of the numpy-converted CS data: ", str(cs_data.shape))

    # Save file to given output DIR 
    ## stem attribute gives the base name of the file without the extension. 
    # For example: If your input file is named sample_data.h5, file.stem will return sample_data
    output_file = os.path.join(save_dir, fname.stem + "_cs.npy")
    np.save(output_file, cs_data)

    # Free up memory and go to next file
    time.sleep(1) 
    del kspace, masked_kspace, mask, cs_data, masked_kspace_ACS, mask_ACS    # Delete the variables to free up memory
    time.sleep(1)
    gc.collect()    # Collect garbage to free up memory
    #print(f"  Saved CS data to {output_file}")
    logging.info(f"  Saved CS data to {output_file}")


    # Timer for the entire file: calculate and print total elapsed time after all slices
    end_time_file = time.time()
    elapsed_time_file = end_time_file - start_time_file
    #print(f"Total time for processing the entire file: {elapsed_time_file:.4f} seconds")
    logging.info(f"Total time for processing the entire file: {elapsed_time_file:.4f} seconds")



def process_dataset_parallel(data_dir, save_dir, mat_file, max_workers=8):
    os.makedirs(save_dir, exist_ok=True)
    ##################### VALIDATION ################################
    # # validation files: 527 brain data, 234 knee 
    # # already taken from knee: 821 from train (still 152 left)
    # # ==> so 152 from train + 82 from val
    # h5_files = list()

    # # Directory containing HDF5 files
    # knee_train_path = Path(data_dir).joinpath("Knee/multicoil_train/")
    # knee_val_path = Path(data_dir).joinpath("Knee/multicoil_val/")
    # brain_val_path = Path(data_dir).joinpath("Preprocessed/multicoil_val/")
    
    # knee_train_files = list(knee_train_path.glob("**/*.h5"))
    # knee_train_files = knee_train_files[-152:]  # Select the last 152 files from the training set
    # h5_files.extend(knee_train_files)

    # knee_val_files = list(knee_val_path.glob("**/*.h5"))
    # knee_val_files = knee_val_files[:82]  # Select the first 82 files from the validation set
    # h5_files.extend(knee_val_files)

    # brain_val_files = list(brain_val_path.glob("**/*.h5"))
    # brain_val_files = brain_val_files[:527]  # Select the first 527 files from the validation set
    # h5_files.extend(brain_val_files)
    # #print("Number of files to process: ", str(len(h5_files)))

    ##################### TEST KNEE: also use validation script #######################
    # Select last 117 files from val set
    knee_val_path = Path(data_dir).joinpath("Knee/multicoil_val/")
    h5_files = list(knee_val_path.glob("**/*.h5"))
    h5_files = h5_files[-117:]  # Select the last 117 files from the validation set
    logging.info(f"Number of files to process: {len(h5_files)}")


    process_fn = partial(process_volume, save_dir=save_dir, mat_file=mat_file)

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
    logging.basicConfig(filename='preprocessing_val.log', level=logging.INFO)
    logging.info('Started processing')
    # Load configuration
    # assumes the config file is named rss_full_config.yaml 
    # and is in the same directory as your script.
    config = load_config("preprocess_val_config.yml") 

    # Use values from config
    data_dir = config["data_dir"]
    save_dir = config["save_dir"]
    mat_dir = config["mat_dir"]
    mat_file = sio.loadmat(mat_dir)
    workers = config["workers"]
    #amount_training_files = config["amount_training_files"]

    start = time.time()
    process_dataset_parallel(data_dir, save_dir, mat_file, max_workers=workers)
    logging.info('Total time for all files: {:.2f} seconds'.format(time.time() - start))
    logging.info('Finished processing')

if __name__ == "__main__":
    main()

