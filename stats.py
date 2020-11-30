from argparse import ArgumentParser
from pytorch_lightning import Trainer
from numpy import mean, std
import os, gc
from classifier import Classifier
import warnings

warnings.filterwarnings("ignore")  # to remove a userwarning from torch_sparse

parser = ArgumentParser()
parser.add_argument("path", help="path directory containing models")
args = parser.parse_args()


def get_test_valid(model_path):
    model = Classifier.load_from_checkpoint(
        checkpoint_path=model_path
    )
    trainer = Trainer(gpus=1, logger=False, weights_summary=None)
    test = trainer.test(model, model.test_dataloader(), verbose=False)
    valid = trainer.test(model, model.val_dataloader(), verbose=False)
    del model
    del trainer
    gc.collect()
    return *test, *valid


test = {}
valid = {}
paths = []
for dir_path, _, file_paths in os.walk(args.path):
    for model_name in file_paths:
        paths.append(os.path.join(dir_path, model_name))

for i, path in enumerate(paths):
    print("evaluating model {}/{}".format(i + 1, len(paths)))
    t, v = get_test_valid(path)
    for k in t.keys():
        if k in test:
            test[k].append(t[k])
            valid[k].append(v[k])
        else:
            test[k] = [t[k]]
            valid[k] = [v[k]]

# .split('/')[0] removes '/valid'
test = {k.split('/')[0]: "{0:.4f} +/- {1:.4f}".format(mean(v), std(v)) for k, v in test.items()}
valid = {k.split('/')[0]: "{0:.4f} +/- {1:.4f}".format(mean(v), std(v)) for k, v in valid.items()}

print("\n\nRESULTS")
for (k, vt), (_, vv) in zip(test.items(), valid.items()):
    print('{0}: TEST {1}  VALID {2}'.format(k, vt, vv))
