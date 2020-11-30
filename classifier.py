from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

from dataset import CombinedOGBEvaluator, CombinedOGBDataset
from operations import ClassifierNetwork

ROOT = 'dataset'

class Classifier(LightningModule):
    def __init__(self, dataset="molpcba", batch_size=100, hidden=100, lr=0.001,
                 layers=3, dropout=0.5, virtual_node=False,
                 conv_radius=3, conv_type='gin+', **kwargs):
        assert conv_type in ['gcn', 'gin', 'gin+', 'naivegin+']
        super().__init__()
        self.save_hyperparameters()
        # Trainer parameters
        self.dataset = dataset
        self.hidden = hidden
        self.lr = lr
        self.batch_size = batch_size

        # Network
        out_dim = self.__dataset__.num_tasks
        self.network = ClassifierNetwork(hidden=hidden,
                                         out_dim=out_dim,
                                         layers=layers,
                                         dropout=dropout,
                                         virtual_node=virtual_node,
                                         k=conv_radius,
                                         conv_type=conv_type)

        # Loss and metrics
        self.evaluator = CombinedOGBEvaluator(name=self.dataset)
        self.metric = self.evaluator.eval_metric
        self.loss_fun = nn.BCEWithLogitsLoss(reduction='none')

    def loss(self, y_pred, y_true):
        y_available = ~torch.isnan(y_true)
        loss = self.loss_fun(y_pred[y_available], y_true[y_available])
        loss = loss.mean()
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-l', '--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('-b', '--batch-size', type=int, default=100, help='batch size')
        parser.add_argument('-d', '--dataset', type=str, default="molpcba", help='ogb dataset or combination of such')
        parser.add_argument('-H', '--hidden', type=int, default=100, help='hidden dimension (num of features')
        parser.add_argument('-L', '--layers', type=int, default=3, help='number of layers (conv blocks)')
        parser.add_argument('-D', '--dropout', type=float, default=0.5, help='dropout rate')
        parser.add_argument('-V', '--virtual-node', action='store_true', help='adds a virtual node')
        parser.add_argument('-K', '--conv-radius', type=int, default=3,
                            help='radius of the conv kernel')
        parser.add_argument('--conv-type', choices=['gcn', 'gin', 'gin+', 'naivegin+'], default='gin+',
                            help='convolution type')
        return parser

    def forward(self, batch):
        return self.network(batch)

    ##########################
    # train, val, test, steps
    ##########################
    def training_step(self, batch, batch_idx):
        y_true = batch.y.float()
        y_pred = self.forward(batch)
        loss = self.loss(y_pred, y_true)
        result = pl.TrainResult(loss)
        result.log('loss/train', loss)
        return result

    def validation_step(self, batch, batch_idx):
        y_true = batch.y.float()
        y_pred = self.forward(batch)
        loss = self.loss(y_pred, y_true)
        result = pl.EvalResult()
        result.loss = loss
        result.pred = y_pred
        result.true = y_true
        result.dataset_idx = batch.dataset_idx
        return result

    def validation_epoch_end(self, validation_step_outputs):
        y_pred = validation_step_outputs.pred
        y_true = validation_step_outputs.true
        dataset_idx = validation_step_outputs.dataset_idx
        loss = validation_step_outputs.loss
        input_dict = {"y_true": y_true, "y_pred": y_pred, "dataset_idx": dataset_idx}
        metrics = {'loss/valid': loss.mean()}
        for k, v in self.evaluator.eval(input_dict).items():
            metrics[k + '/valid'] = v
        result = pl.EvalResult(checkpoint_on=torch.tensor(metrics[self.metric + '/valid']))
        result.log_dict(metrics)
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    ##################
    # Optimizers
    ##################
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    ##################
    # Dataloader
    ##################
    def __make_ds__(self):
        if not hasattr(self, '_dataset'):
            self._dataset = CombinedOGBDataset(root=ROOT, name=self.dataset)
            self._split = self._dataset.get_idx_split()

    @property
    def __dataset__(self):
        self.__make_ds__()
        return self._dataset

    @property
    def __split__(self):
        self.__make_ds__()
        return self._split

    def train_dataloader(self):
        return DataLoader(self.__dataset__[self.__split__["train"]],
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.__dataset__[self.__split__["valid"]],
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.__dataset__[self.__split__["test"]],
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=8)
