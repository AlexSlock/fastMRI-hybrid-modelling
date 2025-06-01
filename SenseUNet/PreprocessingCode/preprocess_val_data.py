import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFunc
import torch
import time
import gc
import bart
import random
import os
import logging

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import json

# nohup python preprocess_val_data.py > output_val.log 2>&1 &

def process_wrapper(args):
    fname, acceleration, save_dir = args
    return process_volume(fname, save_dir, acceleration)

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
    masked_kspace = T.tensor_to_complex_np(masked_kspace_T)
    return masked_kspace, mask

def generate_array(shape, R, tensor_out):
    length = shape[-1]

    # Initialize an array of zeros
    array = np.zeros(length)

    # Determine the central index
    array[length // 2] = 1

    # Set every R-1'th sample to 1, starting from the central index
    for i in range(length // 2, length, R):
        array[i] = 1

    # Mirror the behavior to the first half of the array
    if length % 2 == 0:
        array[1:length // 2] = np.flip(array[length // 2 + 1:])
    else:
        array[:length // 2] = np.flip(array[length // 2 + 1:])

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

def CG_SENSE(kspace, S, lamda=0.005, num_iter=50):
    ''' 
    Performs CG-SENSE reconstruction, i.e. CS reconstruction with a regular l2 norm for which the objective function then corresponds to a SENSE reconstruction.
    https://colab.research.google.com/github/mrirecon/bart-workshop/blob/master/mri_together_2023/bart_mritogether_2023.ipynb#scrollTo=kNWQGBaX9ISp

    Args:
        kspace (numpy.array): Slice kspace of shape (num_coils, rows, cols)
        S (numpy.array): Estimated sensitivity maps given by ESPIRiT of shape (num_coils, rows, cols)
        lamda: Value of the hyperparameter / regularizer of the l2 norm term
        num_iter: The amount of iterations the algorithm can run
    Returns:
        reconstruction (numpy.array): Estimated CG-SENSE reconstruction of shape (rows, cols))
    '''
    # Move coil axis to the back as expected by BART
    kspace_perm = np.moveaxis(kspace, 0, 2)
    S_perm = np.moveaxis(S, 0, 2)
    # Add extra dimension, because BART expects a 4D input array where the third dimension represents the batch size.
    kspace_perm = np.expand_dims(kspace_perm, axis=2)
    S_perm = np.expand_dims(S_perm, axis=2)
    # Perform CG-SENSE reconstruction
    reconstruction = bart.bart(1, 'pics -S -l2 -r {} -i {} -d 0'.format(lamda, num_iter), kspace_perm, S_perm)
    return reconstruction

def process_volume(fname, save_dir, acceleration):
    # Timer for the entire file
    start_time_file = time.time()   

    # Open HDF5 file in read mode
    with h5py.File(fname, 'r') as hf:
        kspace = hf['kspace'][()]

    
    # undersampling_bool = fifty_fifty() # R=4 or R=8

    # # mask for ACS
    # if undersampling_bool:
    #     mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4])
    # else:
    #     mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[8])
    # masked_kspace_ACS, mask_ACS = apply_mask(kspace, mask_func)
    # #print("Shape of the generated ACS mask: ", str(mask_ACS.shape))

    if acceleration == 4:
        mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4])
    else:
        mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[8])
    masked_kspace_ACS, mask_ACS = apply_mask(kspace, mask_func)

    mask = generate_array(kspace.shape, acceleration, tensor_out=False)

    # # mask for SENSE
    # if undersampling_bool:
    #     mask = generate_array(kspace.shape, 4, tensor_out=False)
    # else:
    #     mask = generate_array(kspace.shape, 8, tensor_out=False)
    masked_kspace = kspace * mask + 0.0
    # # print("Shape of the generated SENSE mask: ", str(mask.shape))

    # compute SENSE reconstruction
    sense_data = np.zeros((kspace.shape[0], kspace.shape[2], kspace.shape[3]), dtype=np.complex64)
    for slice in range(kspace.shape[0]):
        S = estimate_sensitivity_maps(masked_kspace_ACS[slice,:,:,:])
        sense_data[slice,:,:] = CG_SENSE(masked_kspace[slice,:,:,:], S)
    # print("Shape of the numpy-converted sense data: ", str(sense_data.shape))

    # Save the Sense data to an HDF5 file
    output_file = Path(save_dir) / fname.name
    with h5py.File(output_file, 'w') as hf_out:
        hf_out.create_dataset('reconstruction', data=sense_data)

    time.sleep(1)
    del kspace, masked_kspace, mask, sense_data, masked_kspace_ACS, mask_ACS
    gc.collect()
    time.sleep(1)

    logging.info(f"  Saved Sense data to {output_file}")

    # Timer for the entire file: calculate and print total elapsed time after all slices
    end_time_file = time.time()
    elapsed_time_file = end_time_file - start_time_file
    #print(f"Total time for processing the entire file: {elapsed_time_file:.4f} seconds")
    logging.info(f"Total time for processing the entire file: {elapsed_time_file:.4f} seconds")

    return (fname.name, acceleration)



def process_dataset_parallel(data_dir, save_dir, max_workers=8):
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
    args_list = [(fname, acc, save_dir) for fname, acc in files_with_af]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_wrapper, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Sense Gen"):
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

    # process_fn = partial(process_volume, save_dir=save_dir)

    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(process_fn, h5_path): h5_path for h5_path in h5_files}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Gen"):
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             logging.error(f"Error processing {futures[future]}: {e}")

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
    workers = config["workers"]
    #amount_training_files = config["amount_training_files"]

    start = time.time()
    process_dataset_parallel(data_dir, save_dir, max_workers=workers)
    logging.info('Total time for all files: {:.2f} seconds'.format(time.time() - start))
    logging.info('Finished processing')

if __name__ == "__main__":
    main()



    

