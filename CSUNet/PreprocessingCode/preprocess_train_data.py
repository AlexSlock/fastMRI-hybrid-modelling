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

# matlab file with CS sampling profiles
mat_file = sio.loadmat('/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/GitLab/master_thesis/fastMRI/sampling_profiles_CS.mat')
# Folder for preprocessed data
folder_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/multicoil_train/'
# Get all files in the folder
files = Path(folder_path).glob('**/*')
file_count = 1

def fifty_fifty():
    '''
    Return True with a probability of 0.5, otherwise False
    '''
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
    masked_kspace_T, mask = T.apply_mask(slice_kspace_T, mask_func)
    # Convert masked k-space from Torch tensor back to numpy array
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

for file in files:
    print(str(file_count)+". Starting to process file "+str(file)+'...')
    undersampling_bool = fifty_fifty()  
    hf = h5py.File(file, 'a') # Open in append mode
    kspace = hf['kspace'][()] 
    print("Shape of the raw kspace: ", str(np.shape(kspace)))
    
    # Randomly decide if R = 4 or 8 for equispaced mask => ACS region for estimating coil sensitivities! 
    if undersampling_bool:
        mask_func = EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4])
    else:
        mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[8])
    masked_kspace_ACS, mask_ACS = apply_mask(kspace, mask_func)
    print("Shape of the generated ACS mask: ", str(mask_ACS.shape))

    # same random if R = 4 or 8 for CS mask
    if undersampling_bool:
        mask = generate_array(kspace.shape, 4, mat_file, tensor_out=False)
    else:
        mask = generate_array(kspace.shape, 8, mat_file, tensor_out=False)
    # (following = OK, see Transforms.apply_mask)
    masked_kspace = kspace * mask + 0.0   # +0.0 removes the sign of the zeros
    print("Shape of the generated CS mask: ", str(mask.shape))

    # Perform CS reconstruction
    cs_data = np.zeros((kspace.shape[0], kspace.shape[2], kspace.shape[3]), dtype=np.complex64)
    for slice in range(kspace.shape[0]):
        S = estimate_sensitivity_maps(masked_kspace_ACS[slice,:,:,:]) # estimate Si with ACS region
        cs_data[slice,:,:] = CS(masked_kspace[slice,:,:,:], S)
    print("Shape of the numpy-converted CS data: ", str(cs_data.shape))

    # Store cs reconstruction in file
    if 'cs_data' in hf:   # Check if 'cs_data' key already exists in the h5 file
        del hf['cs_data'] # Delete the existing dataset
    hf.create_dataset('cs_data', data=cs_data)  # Add a key to the h5 file with cs_data inside it
    hf.close()  # Close the file

    # Free up memory and go to next file
    time.sleep(1)
    del kspace, masked_kspace, mask, cs_data    # Delete the variables to free up memory
    time.sleep(1)
    gc.collect()    # Collect garbage to free up memory
    file_count += 1
    print('Done.')

