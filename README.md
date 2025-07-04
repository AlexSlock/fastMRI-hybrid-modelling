# Master's Thesis - Alex Slock
Code base of my master's thesis (2024-2025), submitted for the degree of M.Sc. in Biomedical Engineering, option Bio-informatics & AI.


Title: Deep Learning-Based MRI Reconstruction
in Data Heterogeneity: A Hybrid Modelling
Approach.


Example visualisation for the CS reconstruction paradigm at R=4:
![plot](./recons_all_R4.png)


## Used dataset
In this thesis, the fastMRI dataset is used. This dataset can be obtained at the official [fastMRI website](https://fastmri.med.nyu.edu/). Note, however, that the hybrid models require that their corresponding preprocessing scripts have already ran.


## Contributions
During the past year, the generalization of the CS UNet hybrid model was analyzed on heterogeneous data. Ultimately, the code in this repository is related to five things:
- fastMRI data exploration
- Evaluation metric implementations (metrics are compared and a new metric is introduced)
- Validation of [DeepMRIRec](https://github.com/stjude/DeepMRIRec/tree/main)
- Hybrid Neural Network comparisons 
- Generalization of the CS UNet hybrid model (main focus of this work)


## Repository structure
Related to the above mentioned explored topics, this repository contains multiple subfolders:
- BaselineUNet
- CS
- CSUNet
- DeepMRIRec
- fastMRI
- Grappa
- GrappaUNet
- ModelWeightsComparison
- Overfitexperiments
- Sense
- SenseUNet
- StatisticalTests
- ZeroFilled (refers to ZF with GRAPPA mask)
- ZeroFilledNoACS (refers to ZF with SENSE mask)
- ZeroFilledNoACSCS (refers to ZF with CS mask)


The folders BaselineUNet, CS, CSUNet, Grappa, GrappaUNet, Sense, SenseUNet, ZeroFilled, ZeroFilledNoACS, and ZeroFilledNoACSCS contain all the code necessary for the HNN comparisons made in the thesis manuscript. Those folders related to a deep learning model also contain a Checkpoint folder containing the trained model used to generate the presented results. The BaselineUNet and GrappaUNet folders even contain multiple saved checkpoints, related to different training scenarios. For example, there is a BaselineUNet version trained only on R=4, on both R=4+8, on both R=4+8 with L2 regularization, and there is also a BaselineUNet version trained on SENSE masks instead of GRAPPA masks. Additionally, there is code to let BaselineUNet make predictions on R=3 data. Similarly, also GrappaUNet has two checkpoints: one for a regularized training and one for an unregularized training. Other models have one checkpoint, resulting from a regularized training as described in the thesis manuscript. Importantly, each of these aformentioned main folders contains a .txt file. In this file, more info can be found regarding which conda environment to use and regarding which source material was used to write the code. Folders containing the code of Hybrid Neural Networks also contain the code that was used to preprocess the fastMRI data up front, before training, saving time during training.

The CSUnet has 3 checkpoints: one for a regularized training on only brain data, one for unregularized training on only brain data, and different from the other models: one for regularized training on both knee and brain data.

The folders fastMRI, ModelWeightsComparison, Overfitexperiments, and StatisticalTests are all related to data and results analyses. These mostly contain notebooks, with their content and related comments speaking for itself.


Lastly, the folder DeepMRIRec contains the used code for the training of DeepMRIRec. This folder contains the code for multiple variations of its architecture, and multiple variations of the used dataloader. When wanting to run the model, the small GPU version of the model might need to be used, as the original version is quite heavy.


## Creating correct conda environments
Throughout this repository, three different conda environments are used. In order to make the code as plug-and-play as possible, these conda environments were exported to .yml files and inserted in this repository. Additionally, a .txt file was created indicating which python versions to use.


## Training the HNNs
This is done through the commandline, for example:
- Go to the folder of the desired HNN
- Activate the correct conda env, listed in the .txt file
- Run `python UNet.py --mode train --challenge multicoil --mask_type equispaced --center_fractions 0.08 0.04 --accelerations 4 8`


Configs for where to save checkpoints and metadata can be changed in the folder's yaml file.


## HNN model inference
This is done through the commandline, for example:
- Go to the folder of the desired HNN
- Activate the correct conda env, listed in the .txt file
- Run `python UNet.py --mode test --challenge multicoil --mask_type equispaced --resume_from_checkpoint checkpoint-path`
- Change checkpoint-path to the correct path


Configs for where to save reconstructions made on the test dataset can be changed in the folder's yaml file.


## HNN inference evaluations
This is done through the commandline, for example:
- Go to the folder of the desired HNN
- Activate the correct conda env, listed in the .txt file
- Run `python evaluate_with_vgg_and_mask.py --target-path path1 --predictions-path path2 --challenge multicoil --acceleration 4`
- Change path1 and path2 to the correct paths


The results will be printed in the terminal.


## Other examples
Other examples of how to use the commandline to run the code can often be found on the repositories linked in the .txt files when necessary. Alternatively, the required input arguments to run a python file, and what they do, can also be deduced by looking at the code. The code is very well-structured, and often contains comments when necessary.


## Training DeepMRIRec
The files pertaining to the validation of DeepMRIRec can be found in the DeepMRIRec folder. Make sure you use a GPU when running a DeepMRIRec training process. The micgpu command only ensures that if you use GPU, this is done on the desired GPU number. Luckily, when using tensorflow, models will transparently run on a single GPU with no code changes required if the tensorflow's required CUDA and cuDNN versions are installed and added to the PATH and LD_LIBRARY_PATH variables. If this is not the case, CUDA will throw an error and start using a CPU. The following set-up is known to work for tensorflow 2.2.0, used by the DL_MRI_reconstruction conda environment compatible with the code of DeepMRIRec:
- Activate the conda env DL_MRI_reconstruction
- In your terminal run `source cuda_settings.sh`


Note that this is not necessary for the HNNs' code, as their conda environments automatically handle GPU usage there.


## Reference
When using anything of this work, please refer to:
```
Slock A. (2025). Deep Learning-Based MRI Reconstruction in Data Heterogeneity: A Hybrid Modelling Approach. Availabe on: https://github.com/AlexSlock/fastMRI-hybrid-modelling/
[accessed on ...].
```
and previous work this code is based on:
```
Vanhaverbeke M. (2024). Influence of hybrid modelling on deep learning-based MRI reconstruction performance. Available on: https://github.com/MathijsVanhaverbeke/fastMRI-hybrid-modelling/ [accessed on ...].
```


