### VERY IMPORTANT CONTRAST WITH DeepMRIRec: DEEPMRIREC IS AN IMAGE SPACE METHOD


## Set up a resource limiter, such that the script doesn't take up more than a certain amount of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

resource.setrlimit(resource.RLIMIT_AS, (40_000_000_000, 40_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import random, h5py
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy import fft 
import os
from pygrappa import grappa
import gc
import time 
from pathlib import Path
from itertools import chain
import imgaug.augmenters as iaa


print('Libraries imported. Starting to prepare to load the dataset...')


## Prepare dataset

download_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI'
dicom_path = os.path.join(download_path,'fastMRI_brain_DICOM')
train_path = os.path.join(download_path,'multicoil_train')
validation_path = os.path.join(download_path,'multicoil_val')
test_path = os.path.join(download_path,'multicoil_test')
fully_sampled_test_path = os.path.join(download_path,'multicoil_test_full')

training_files = Path(train_path).glob('**/*')
validation_files = Path(validation_path).glob('**/*')
test_files = Path(test_path).glob('**/*')
fully_sampled_test_files = Path(fully_sampled_test_path).glob('**/*')
all_files = chain(training_files, validation_files, test_files, fully_sampled_test_files)

clustered_data_2 = np.load("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/exploration/16coil_slice_size_clustered_fastmri_data.npy", allow_pickle=True)
clustered_data_2 = clustered_data_2.item()

files_16_640_320 = clustered_data_2[(640,320)]
training_files = sorted(files_16_640_320)

crop_size = (12,640,320)


print('All variables are loaded. Start preprocessing data in batches...')



training_batch_number = 1
validation_batch_number = 1
batch_size = 1
# Note: batch_size here has unit 'number of files', not 'number of slices'



## Loop across training data to preprocess in batches

# First, we define helper functions
def calculate_mask(mask,start,end,nPE):
    '''
    The mask contains more information from the center of k-space. 
    We divide phase encoding space into nine areas/slots (see Figure 3 from paper, nine slots are separated by red dotted lines) 
    and select 10% of data points from the center area and 4%, 2%,1%, and 0.5% from the area adjacent to the centers
    In total, we thus retain 10% + (4+2+1+0.5)*2 % = 25% of the original k-space. This way, we obtain R=4.
    '''
    
    total_point = start
    max_len = nPE
    number_of_sample = (max_len/4)-(end-start)
    step = int((4*start)/number_of_sample)
    i = step
    f = 1
    indx = 1
    while i < total_point+1:
        offset = 10 + int(random.sample(range(0,3), 1)[0])
        if offset+i < total_point:
            mask[:,:,:,offset+i] = 1
        else:
            mask[:,:,:,i] = 1

        offset = 5 + int(random.sample(range(0,3), 1)[0])
        if i-offset > 0:
            mask[:,:,:,max_len-i-offset] = 1
        else:
            mask[:,:,:,max_len-i] = 1
        i = i+step
        indx = indx+1
        if i >= ((total_point*f)//2):
            step = int(step/3)
            f = f+1
    return mask

def Grappa_recon(kspace,start,end):
    calib = kspace[:,:,start:end].copy()
    res = grappa(kspace, calib, kernel_size=(5,5),coil_axis=0)
    return res

def comp_img(img,crop_size):
    ## Note: only functions correctly if the height and width of a slice of img are larger than the desired crop size
    s = img.shape
    start_height = s[1]//2 - (crop_size[1]//2)
    start_width = s[2]//2 - (crop_size[2]//2)
    return img[:,start_height:(start_height+crop_size[1]),start_width:(start_width+crop_size[2])]

def select_slices(array, slice_index, total_slices):
    # Ensure the array has 4 dimensions
    if array.ndim != 4:
        raise ValueError("Input array must be 4-dimensional")
        
    # Ensure total_slices is not greater than the size of the first dimension
    if total_slices > array.shape[0]:
        raise ValueError("Total slices must be less than or equal to the size of the first dimension of the input array")
    
    # All slices before slice_index are definitely kept
    slices_to_keep = array[:slice_index, :, :, :]
    
    # Only select slices beyond slice_index if total_slices is greater than slice_index
    if total_slices > slice_index:
        slices_to_select = array[slice_index:, :, :, :]
        
        # Randomly select the remaining slices to reach total_slices
        indices = np.random.choice(slices_to_select.shape[0], total_slices - slice_index, replace=False)
        slices_to_add = slices_to_select[indices, :, :, :]
        
        # Concatenate the slices to keep and the newly selected slices
        output_array = np.concatenate((slices_to_keep, slices_to_add), axis=0)
    else:
        output_array = slices_to_keep

    return output_array

# Define possible image augmentations
seq = iaa.Sequential([

    iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Dropout([0.1, 0.2]),
    iaa.GaussianBlur(sigma=(1.5)),
    iaa.GaussianBlur(sigma=(0.8)),
    iaa.GaussianBlur(sigma=(3)),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.4, 3.5)),

    #PiecewiseAffine and elastic deformation
    iaa.PiecewiseAffine(scale=(0.01, 0.03)),
    iaa.PiecewiseAffine(scale=(0.04, 0.06)),
    iaa.ElasticTransformation(alpha=(2.0,4.0), sigma=1),
    iaa.ElasticTransformation(alpha=(14.0,17.0), sigma=5),
    iaa.Affine(rotate=(-20, -15),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (0.8), "y": (0.8)}, order=3,
            ),
    iaa.Affine(rotate=(-10, -5),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (1.1), "y": (1.1)}, order=3,
            ),
    iaa.Affine(rotate=(5, 10),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (0.8), "y": (0.8)}, order=3,
            ),
    iaa.Affine(rotate=(15, 20),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (1.1), "y": (1.1)}, order=3,
            )
              
], random_order=False)

def apply_augmentation(seq_picked,image,ref):
    augseq_det = seq_picked.to_deterministic()
    dist_image = augseq_det.augment_image(image)
    ref_image = augseq_det.augment_image(ref)
    return dist_image, ref_image

# Now, we start the training data construction and GRAPPA image reconstruction estimation

num_batches = len(training_files) // batch_size

for batch in range(num_batches):

    batch_files = training_files[batch*batch_size:(batch+1)*batch_size]

    print('Preprocessing training batches '+str(training_batch_number)+'-'+str(training_batch_number+3)+' and validation batch '+str(validation_batch_number)+'...')


    ## Create training data pairs and estimate GRAPPA kernels

    print('Starting training dataset construction and initial GRAPPA image reconstructions of the undersampled data...')

    cnt = 1
    last_mask = None
    X_train = []
    Y_train = []
    for mri_f in batch_files:
        filename = os.path.basename(mri_f)
        filename = filename.replace(".h5","")
        with h5py.File(mri_f,'r') as f:

            k = f['kspace'][()]
            sequence = f.attrs['acquisition']
            nSL, nCh, nFE, nPE = k.shape
            
            # Select ACS region
            mid = nPE//2
            start = mid-int(nPE*0.05)
            end = mid+int(nPE*0.05)
            
            mask = 0*k
            mask[:,:,:,start:end] = 1
            mask = calculate_mask(mask,start,end,nPE)  
            last_mask = mask
            subsampled_k = k*mask
            ts = time.time()
            if nCh > 18:
                channels_picked = [0,1,2,4,7,8,11,13,15,16,17,18]
            elif nCh > 12 and nCh < 18:
                channels_picked = [0,1,2,3,4,5,6,7,8,10,11,12]
            else:
                channels_picked = []
            if len(channels_picked)>0:
                for slices in range(nSL): 
                    chindx = 0
                    ref_img = np.zeros((len(channels_picked),nFE,nPE),dtype=np.float32)
                    sub_img = np.zeros((len(channels_picked),nFE,nPE),dtype=np.float32)
                    subsampled_tmp = Grappa_recon(subsampled_k[slices,:,:,:],start,end)    
                    for iCh in channels_picked:
                        sub_img[chindx,:,:] = abs(fft.fftshift(fft.ifft2(subsampled_tmp[iCh,:,:])))
                        ref_img[chindx,:,:] = abs(fft.fftshift(fft.ifft2(k[slices,iCh,:,:])))
                        chindx = chindx+1
                    X_train.append(list(comp_img(sub_img,crop_size)))
                    Y_train.append(list(comp_img(ref_img,crop_size)))
                print(cnt,filename,sequence,nSL,nCh,nFE,nPE,sub_img.shape,"ACS region indices: ",start,end,"Processing time: ", time.time()-ts)
                cnt += 1
                gc.collect()
                time.sleep(1)
    X_train_arr = np.array(X_train).astype(np.float32)
    Y_train_arr = np.array(Y_train).astype(np.float32)
    minimum_slices = X_train_arr.shape[0]


    print('Done. Applying image augmentations...')


    # Apply image augmentation

    augmented_image_X = []
    augmented_image_Y = []
    X_train_arr = np.transpose(X_train_arr,(0,2,3,1))
    Y_train_arr = np.transpose(Y_train_arr,(0,2,3,1))
    for k in range(X_train_arr.shape[0]):
        image = X_train_arr[k,:,:,:].astype(np.float32)
        ref = Y_train_arr[k,:,:,:].astype(np.float32)
        augmented_image_X.append(image)
        augmented_image_Y.append(ref)
        for i in range(len(seq)):
            seq_picked = iaa.Sequential([seq[i]])
            dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
            augmented_image_X.append(dist_image)
            if i in [2,3,4,5,6]:
                augmented_image_Y.append(ref)
            else:
                augmented_image_Y.append(ref_image)
            if dist_image.shape[0] != crop_size[1] and dist_image.shape[1] != crop_size[2] and dist_image.shape[1] != crop_size[0]:
                print("Warning: augmentation resulted in a different slice size: ",image.shape,dist_image.shape,ref_image.shape)
        
        #PiecewiseAffine and Affine
        seq_picked = iaa.Sequential([seq[7],seq[11]])
        dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
        augmented_image_X.append(dist_image)
        augmented_image_Y.append(ref_image)
        
        #Elastic and Affine
        seq_picked = iaa.Sequential([seq[10],seq[11]])
        dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
        augmented_image_X.append(dist_image)
        augmented_image_Y.append(ref_image)
        
        seq_picked = iaa.Sequential([seq[9],seq[14]])
        dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
        augmented_image_X.append(dist_image)
        augmented_image_Y.append(ref_image)

    X_train_arr = np.array(augmented_image_X)
    Y_train_arr = np.array(augmented_image_Y)
    X_train_arr = select_slices(X_train_arr, minimum_slices, 36)
    Y_train_arr = select_slices(Y_train_arr, minimum_slices, 36)
    # Optimal training mini-batch size is apparently 32: https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
    # 90% of 36 = 32
    # 10% of 36 = 4


    print('Done. Calculating RSS images as references for the loss function...')


    ## Calculate RSS images that will be used as references for the loss function

    Y_rss = np.sqrt(np.sum(np.square(Y_train_arr),axis=3))
    Y_rss = Y_rss.astype(np.float32)


    print('Done. Normalizing the data...')


    ## Normalize the data

    def normalize(I):
        mn = I.min()
        mx = I.max()

        mx -= mn

        #I = ((I - mn)/mx) * 255.0
        #return np.round(I).astype(np.uint8)
    
        I = ((I - mn)/mx)
        return I.astype(np.float32)

    dims = X_train_arr.shape

    for i in range(dims[0]):
        for j in range(dims[3]):
            X_train_arr[i,:,:,j] = normalize(X_train_arr[i,:,:,j])

    for i in range(dims[0]):
        Y_rss[i,:,:] = normalize(Y_rss[i,:,:])
    
    #X_train_arr = X_train_arr.astype(np.uint8)
    #Y_rss = Y_rss.astype(np.uint8)


    print('Done. Performing a datasplit...')


    ## Create a dataset split 90-10 training-validation

    Y_rss = Y_rss.reshape((dims[0],dims[1],dims[2],1))
    x_train = X_train_arr[0:int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1),:,:,:]
    y_train = Y_rss[0:int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1),:,:,:]
    x_test = X_train_arr[int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1):,:,:,:]
    y_test = Y_rss[int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1):,:,:,:]


    print('Done. Creating sub-batches for the training data...')


    ## Create smaller sub-batches for the training data

    def sub_batch(array1, array2):

        dim_size = array1.shape[0]
        indices = np.random.choice(dim_size, size=dim_size, replace=False)
        selection_size = len(indices) // 4
        indices_1 = list(indices[0:selection_size])
        indices_2 = list(indices[selection_size:2*selection_size])
        indices_3 = list(indices[2*selection_size:3*selection_size])
        indices_4 = list(indices[3*selection_size:4*selection_size])

        array1_1 = array1[indices_1]
        array1_2 = array1[indices_2]
        array1_3 = array1[indices_3]
        array1_4 = array1[indices_4]

        array2_1 = array2[indices_1]
        array2_2 = array2[indices_2]
        array2_3 = array2[indices_3]
        array2_4 = array2[indices_4]

        return array1_1, array1_2, array1_3, array1_4, array2_1, array2_2, array2_3, array2_4
    
    x_train_1, x_train_2, x_train_3, x_train_4, y_train_1, y_train_2, y_train_3, y_train_4 = sub_batch(x_train, y_train)


    print('Done. Saving results...')


    ## Save the results

    path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri_augmented_float32_small_batch/'

    np.save(path_to_save_mri_data+"training_data_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number), x_train_1)
    np.save(path_to_save_mri_data+"training_data_GT_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number), y_train_1)
    np.save(path_to_save_mri_data+"training_data_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number+1), x_train_2)
    np.save(path_to_save_mri_data+"training_data_GT_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number+1), y_train_2)
    np.save(path_to_save_mri_data+"training_data_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number+2), x_train_3)
    np.save(path_to_save_mri_data+"training_data_GT_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number+2), y_train_3)
    np.save(path_to_save_mri_data+"training_data_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number+3), x_train_4)
    np.save(path_to_save_mri_data+"training_data_GT_DeepMRIRec_16_coils_batch_{}.npy".format(training_batch_number+3), y_train_4)
    np.save(path_to_save_mri_data+"validation_data_DeepMRIRec_16_coils_batch_{}.npy".format(validation_batch_number), x_test)
    np.save(path_to_save_mri_data+"validation_data_GT_DeepMRIRec_16_coils_batch_{}.npy".format(validation_batch_number), y_test)
    

    print('Done. Clearing memory and preparing to load in a new batch...')


    ## Free up RAM memory

    time.sleep(1)
    del X_train
    del Y_train
    del augmented_image_X
    del augmented_image_Y
    del X_train_arr
    del Y_train_arr
    del Y_rss
    time.sleep(1)

    gc.collect()


    print('Done.')
    training_batch_number += 4
    validation_batch_number += 1



## Visualize an example of the processed data

# Slice
indx = 3

fix,ax = plt.subplots(nrows=1,ncols=2,figsize=(6,10))
ax[0].imshow(x_train[indx,:,:,0],cmap='gray')
ax[1].imshow(y_train[indx,:,:],cmap='gray')

plt.show()


print('Done.')

