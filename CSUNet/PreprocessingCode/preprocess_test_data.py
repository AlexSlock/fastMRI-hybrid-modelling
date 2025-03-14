import h5py
import numpy as np
from pathlib import Path
from fastmri.data import transforms as T
import torch
import time
import gc
import bart
import scipy.io as sio


mat_file = sio.loadmat('/DATASERVER/MIC/GENERAL/STUDENTS/aslock2/fastMRI-hybrid-modelling/fastMRI/sampling_profiles_CS.mat')
# Select test data folder
folder_path = '/DATASERVER/MICS/SHARED/NYU_FastMRI/Preprocessed/multicoil_test/'
folder_path_full = '/DATASERVER/MICS/SHARED/NYU_FastMRI/Preprocessed/multicoil_test_full/'
files = Path(folder_path).glob('**/*')
file_count = 1

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
    masked_kspace_T, mask = T.apply_mask(slice_kspace_T, mask_func)
    masked_kspace = T.tensor_to_complex_np(masked_kspace_T)
    return masked_kspace, mask

def generate_array(shape, R, mat_file, tensor_out):
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

def closer_to_4_or_8(float):
    diff_4 = np.abs(float - 4)
    diff_8 = np.abs(float - 8)

    if diff_4 < diff_8:
        return int(4)
    elif diff_8 < diff_4:
        return int(8)

for file in files:
    # For test files: official undersampled versions of data + masked are given as well 
    # (done by retrospective 2D Cartesian undersampling in PE direction)

    print(str(file_count)+". Starting to process file "+str(file)+'...')
    # Use given masks to determine R + look at the shape of the kspace data
    hf = h5py.File(file, 'r') # Open in read mode!
    nPE_mask = hf['mask'][()]
    sampled_columns = np.sum(nPE_mask)
    R = len(nPE_mask)/sampled_columns
    R = float(R)
    masked_kspace_ACS = hf['kspace'][()]
    print("Shape of the raw kspace: ", str(np.shape(masked_kspace_ACS)))

    # calculate CS reconstruction with deriven R and ACS region
    # (open multicoil_test_full file: where original k_space is stored)
    hf = h5py.File(folder_path_full+file.name, 'r') # Open in read mode!
    kspace = hf['kspace'][()]
    mask = generate_array(kspace.shape, closer_to_4_or_8(R), mat_file, tensor_out=False)
    masked_kspace = kspace * mask + 0.0
    cs_data = np.zeros((masked_kspace.shape[0], masked_kspace.shape[2], masked_kspace.shape[3]), dtype=np.complex64)
    for slice in range(masked_kspace.shape[0]):
        S = estimate_sensitivity_maps(masked_kspace_ACS[slice,:,:,:])
        cs_data[slice,:,:] = CS(masked_kspace[slice,:,:,:], S)
    print("Shape of the numpy-converted CS data: ", str(cs_data.shape))

    # Save the CS data to the file
    hf = h5py.File(file, 'a') # Open in append mode!
    # Check if 'cs_data' key exists
    if 'cs_data' in hf:
        del hf['cs_data'] # Delete the existing dataset
    # Add a key to the h5 file with cs_data inside it
    hf.create_dataset('cs_data', data=cs_data)
    hf.close()

    # Clean up
    time.sleep(1)
    del nPE_mask, masked_kspace_ACS, kspace, mask, masked_kspace, cs_data
    time.sleep(1)
    gc.collect()
    file_count += 1
    print('Done.')

