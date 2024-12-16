from setuptools import setup

setup(
    name='fastMRI Prostate',
    version='1.0',
    description='A large scale dataset and reconstruction script of both raw prostate MRI measurements and images',
    packages=["fastmri_prostate", "DL_reconstruction", "fastmri_prostate_classification"], # needed to add this!
    install_requires=[
        'h5py==3.7.0',
        'numpy==1.23.5',
        'scikit-image==0.19.2',
        'torchvision>=0.8.1',
        'torch>=1.8.0',
        'runstats>=1.8.0',
        'pytorch_lightning>=1.4',
        'h5py>=2.10.0',
        'PyYAML>=5.3.1',
        'torchmetrics>=0.5.1',
        'pandas>=1.3.4',
        'opencv-python>=4.5.5', # changed from == to >= 
        'scipy>=1.6.2'
    ],
    python_requires='>=3.9'
)

# also still had to still install pykernel to work with .ipynb => conda install ipykernel