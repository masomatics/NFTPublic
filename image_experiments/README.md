# Neural Fourier Transform: A General Approach to Equivariant Representation Learning

This repository contains the code for the experiments detailed in Section 5.2: "Applications to Images", from our paper. The repository's structure follows the pattern established by the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) project. Configuration files for each experiment are stored in the `config` directory, allowing easy adjustment and tracking of experimental settings. Configurations are formatted in YAML and can be readily viewed by opening the respective files.

## Getting Started

### Installation

Dependencies required for this project can be installed using `pip -r requirements.txt`. 

### Dataset Setup

To perform the 3D rendering experiments, you'll need to download and extract the requisite datasets. The path to each dataset should then be input into the corresponding `config/data/${data}.yaml` files. For instance, the `config/data/abo_material.yaml` file contains the following lines:
```
dataset:
  _target_: src.datamodule.abo.ABO_Material
  root: /path/to/dataset
  ...
```
Replace `/path/to/dataset` with the actual path to your dataset. Some datasets may require additional setup for the creation of metadata files. You can check the details in `src/datamodule/${data}.py`. 

## Running Experiments

### Training

For Roated MNIST experiments, use the following command to train a model:
```
python src/train.py experiment=rot_mnist model=${model} ${additional_options} 
```
Here, `${model}` is a placeholder for the model you wish to train. Current options include `ae, equiv_ae, inv_ae, mspae, supervised, contrastive`, where `equiv_ae` represents g-NFT. The `${additional_options}` placeholder allows for further customization of the learning settings. For instance, the default setting uses all available GPUs, but if you prefer to use the CPU, add `trainer.accelerator=cpu` to your command. 

For the 3D rendering experiments, use the command below:
```
python src/train.py experiment=${experiment} ${additional_options} 
```
In this case, `${experiment}` represents the chosen dataset: either `modelnet`, `complex_brdfs`, or `abo_material`. Please be aware that the 3D rendering experiments demand significant memory. If you run into Out-Of-Memory (OOM), try reducing the batch size by including `data.batch_size=${your_batchsize}` in your command.