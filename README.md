This is a PyTorch implementation of our paper:

[Graph convolutions that can finally model local structure - Rémy Brossard, Oriel Frigo, David Dehaene (2020)](https://arxiv.org/abs/2011.15069)

# Installation
```
git clone https://github.com/RBrossard/GINEPLUS.git
cd GINEPLUS
conda env create --name myenv -f environment.yml
conda activate myenv
```

# Results on ogbg-molpcba

In the paper, we demonstrate the following performance on the ogbg-molpcba dataset from Stanford Open Graph Benchmark (1.2.3)

Model |	Test Accuracy |	Valid Accuracy |	Parameters |	Hardware
----- | ------------- | -------------  | ------------  | -----------
GINE+ | 0.2917 ± 0.0015 | 0.3065 ± 0.0030 | 5,946,128 | GeForce GTX 1080 Ti (11 Go)

## Reproducing results

The command line to reproduce *one time* this experiment is the following:

`python train_classifier.py --name gineplus -V -L 5 -H 400 -K 3 --conv-type gin+  --dataset molpcba -b 100 -l 0.001`

In order to reproduce the statistics, we provide a script `reproduce-ogbg-molpcba.sh` 
that runs the experiment five times and agglomerate statistics using `stats.py`. 
(NB: Due to random initialisation, the numbers might be slightly different, as the random seeds are not controlled.)

```
chmod +x reproduce-ogbg-molpcba.sh
./reproduce-ogbg-molpcba.sh
```

## Detailed hyperparameters

Essential model and training hyperparameters:
```
--batch_size 100    --conv_radius 3     --conv_type gin+ 
--dataset molpcba   --dropout 0.5       --hidden 400 
--layers 5          --lr 0.001          --max_epochs 100 
--virtual_node
```

Other hyperparameters (mainly pytorch-lightning default for training and reporting).

```
--accumulate_grad_batches 1             --amp_backend native            --amp_level O2
--auto_lr_find False                    --auto_scale_batch_size False   --auto_select_gpus False
--benchmark False                       --check_val_every_n_epoch 1     --checkpoint_callback True
--default_root_dir None                 --deterministic False           --distributed_backend None
--early_stop_callback False             --fast_dev_run False            --gpus 1
--gradient_clip_val 0                   --limit_test_batches 1.0        --limit_train_batches 1.0
--limit_val_batches 1.0                 --log_gpu_memory None           --log_save_interval 100
--logger True                           --max_steps None                --min_epochs 1
--min_steps None                        --name gineplus                 --num_nodes 1
--num_processes 1                       --num_sanity_val_steps 0        --overfit_batches 0.0
--overfit_pct None                      --precision 32                  --prepare_data_per_node True
--process_position 0                    --profiler None                 --progress_bar_refresh_rate 1
--reload_dataloaders_every_epoch False  --replace_sampler_ddp True      --resume_from_checkpoint None           
--row_log_interval 50                   --sync_batchnorm False          --terminate_on_nan False
--test_percent_check None               --track_grad_norm -1            --train_percent_check None
--truncated_bptt_steps None             --val_check_interval 1.0        --val_percent_check None
--weights_save_path None                --weights_summary top
```



# Usage
## Training

Execute file `train_classifier.py`

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