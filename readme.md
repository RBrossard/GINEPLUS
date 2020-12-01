This is a PyTorch implementation of our paper:

[Graph convolutions that can finally model local structure - Rémy Brossard, Oriel Frigo, David Dehaene (2020)](Graph convolutions that can finally model local structure)

# Installation
```
conda env create --name myenv -f environment.yml
conda activate myenv
```

# Usage
## Training

Execute file train_classifier.py

We use pytorch-lightning. All arguments available for pytorch_lightning.Trainer object are available, 
such as ``--gpus`` or ``--resume_from_checkpoint``.

Additionnal, model specific argument are the following:


`--name NAME` : name of the experiment used for logging. We use trains from AllegroAI for logging.

`-l --lr LR`: learning rate (default: 0.001)

`-b --batch-size BATCH_SIZE` : batch size (default: 100)

`-d --dataset DATASET`: dataset name (default "molpcba"). See details below.   

`-H --hidden HIDDEN` : embedding dimension of the nodes. (default:100)

`-L --layers LAYERS` : number of convolution layers. (default: 3)

`-D --dropout DROPOUT` : dropout rate (default 0.5)

`-V --virtual-node` : If specified, uses a virtual node.

`-K --conv-radius CONVRADIUS`: Radius of the GINE+ and NaiveGINE+ convolutions. (default: 4)

`--conv-type CONVTYPE` : Type of convolution, must be one of gin+, naivegin+, gin or gcn. (default: gin+)

### Datasets
The code allows any dataset from OGB, with their default split (which is scaffold split on molecule datasets. 
All dataset names will be prefixed with "ogbg-". The split ('scaffold', 'random' or 'default') can be specified using '_'.
If not specified, OGB default split is used. Datasets can be combined using '+'. Example: Tox21 with random split combined 
with PCBA dataset using default split: `--dataset moltox21_random+molpcba`.
If a random split is used for the first time, the split indexes will be saved, so that the next time that the same dataset 
is used with random split (eg. at test time), the same dataset split will be used.

All datasets are saved by default in the folder `dataset` of your project. 
Change line 14: `ROOT = 'dataset'` if you want to change the dataset location.

Example:

`python train_classifier.py --name myexp --dataset moltox21_random+moltoxcast -L 3 -H 100 -V --conv-type gin+`

## Statistics

The training script will save the weights of the model with higher score metric.

We provide a script `stats.py` that takes a directory path as input and will agglomerate the statistics 
of all models in that directory.

`python stats.py /path/to/my/model`