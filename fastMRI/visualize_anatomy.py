from pathlib import Path
import h5py
import numpy as np
import fastmri
from matplotlib import pyplot as plt
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

def visualize_anatomy(train_path, coil_indices, center_fractions, accelerations):
    """
    Visualizes k-space and reconstructions for fully sampled and undersampled data from a given train_path.

    Parameters:
        train_path (str or Path): Path to the training data directory containing .h5 files.
        coil_indices (list): List of coil indices to visualize.
        center_fractions (list): Fraction of low-frequency k-space retained for undersampling.
        accelerations (list): Acceleration factor for undersampling.

    Returns:
        None
    """
    # Ensure train_path is a Path object
    train_path = Path(train_path)

    # Get list of .h5 files
    train_list = list(train_path.glob('*.h5'))
    if not train_list:
        print(f"No .h5 files found in {train_path}")
        return

    print(f"Found {len(train_list)} files in {train_path}")

    ####### visualize fully sampled data

    # Open the first file
    file_name = train_list[0]
    print(f"Using file: {file_name}")
    with h5py.File(file_name, 'r') as hf:
        volume_kspace = hf['kspace'][()]

    print(f"Volume k-space shape: {volume_kspace.shape}")

    # Select the middle slice
    slice_index = volume_kspace.shape[0] // 2
    slice_kspace = volume_kspace[slice_index]

    def show_coils(data, coil_nums, cmap=None):
        fig = plt.figure(figsize=(12, 8))
        for i, num in enumerate(coil_nums):
            plt.subplot(1, len(coil_nums), i + 1)
            plt.imshow(data[num], cmap=cmap)
            plt.title(f"Coil {num}")
        plt.show()

    # Visualize raw k-space for selected coils
    print("Visualizing raw k-space...")
    show_coils(np.log(np.abs(slice_kspace) + 1e-9), coil_indices, cmap='gray')

    # Fully sampled reconstruction of coils
    slice_kspace2 = T.to_tensor(slice_kspace)
    slice_image = fastmri.ifft2c(slice_kspace2)
    slice_image_abs = fastmri.complex_abs(slice_image)

    print("Visualizing fully sampled reconstruction...")
    show_coils(slice_image_abs, coil_indices, cmap='gray')

    # Root-Sum-of-Squares (RSS) reconstruction (all coils together)
    slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
    plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
    plt.title("Fully Sampled RSS Reconstruction")
    plt.show()

    ####### Undersampled data (input of our classical methods/ hybrid models!)

    mask_func = RandomMaskFunc(center_fractions, accelerations)
    masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)

    # plot raw k-space
    print('visualize under-sampled k-space by GRAPPA maks')
    show_coils(np.log(np.abs(fastmri.complex_abs(masked_kspace).numpy()) + 1e-9), coil_indices, cmap='gray')

    # visualize RSS 
    sampled_image = fastmri.ifft2c(masked_kspace)
    sampled_image_abs = fastmri.complex_abs(sampled_image)
    sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

    print("Visualizing undersampled reconstruction...")
    plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
    plt.title("Undersampled RSS Reconstruction")
    plt.show()

    #TODO: add visualization of classical and hybrid models!

# Example usage
# from visualize_anatomy import visualize_anatomy
# visualize_anatomy('/DATASERVER/MIC/SHARED/NYU_FastMRI/Knee/multicoil_train', coil_indices=[0, 5, 10], center_fractions=[0.08], accelerations=[4])
