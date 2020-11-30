from argparse import ArgumentParser, Namespace
import os
import time

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.loggers import TrainsLogger
from pytorch_lightning.core.memory import LayerSummary


from classifier import Classifier

def main(args: Namespace):
    ###############
    # Warning: model must be defined before trains task are created
    # if not, the initial call to build the dataset will call torch.save, and trains will believe many models co-exists
    # this will break the model upload routine.
    ###############
    if args.resume_from_checkpoint is not None:
        model = Classifier.load_from_checkpoint(args.resume_from_checkpoint)
    else:
        model = Classifier(**vars(args))
    summary = LayerSummary(model)
    print("number of parameters:", summary.num_parameters)

    task_name = "{}/{}/{}/".format(args.dataset, args.name, time.strftime("%Y-%m-%d_%Hh%M:%S"))
    trains_logger = TrainsLogger(
        project_name="Classifier",
        task_name=task_name
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'runs', task_name, '{epoch}'),
        save_top_k=1,
        monitor=model.metric+'/valid',
        verbose=True,
        mode='max',
        prefix='',
        period=0,
    )

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=trains_logger,
                                         )
    trainer.fit(model)
    print("##################\nTest performance: \n##################")
    test_results = trainer.test(test_dataloaders=model.test_dataloader())
    print("##################\nValidation performance: \n##################")
    valid_results = trainer.test(test_dataloaders=model.val_dataloader())
    return valid_results, test_results

if __name__ == '__main__':
    parser = ArgumentParser()

    ############
    # PROGRAM ARGS
    ############
    parser.add_argument('--name', type=str, default='classification', help='name in trains')

    ############
    # REAL HYPERPARAMETERS
    ############
    parser = Classifier.add_model_specific_args(parser)

    ############
    # ALL TRAINER ARGS
    ############
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1,
                        max_epochs=100,
                        num_sanity_val_steps=0)
    hparams = parser.parse_args()
    main(hparams)
