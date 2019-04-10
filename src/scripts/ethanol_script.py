import logging
from torch.optim import Adam
import os
from shutil import rmtree
from schnetpack.atomistic import AtomisticModel
from schnetpack.output_modules import Atomwise
from schnetpack.data import AtomsData, AtomsLoader, train_test_split
from schnetpack.representation import SchNet
from schnetpack.train import Trainer, TensorboardHook, CSVHook, ReduceLROnPlateauHook
from schnetpack.metrics import MeanAbsoluteError
from schnetpack.utils import loss_fn
from scripts.script_utils.arg_parsing import get_parser


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

parser = get_parser()
args = parser.parse_args()

# basic settings
data_dir = args.data_path
model_dir = args.model_dir
logging.info('datapath: {}'.format(os.path.abspath(data_dir)))
os.makedirs(data_dir, exist_ok=True)
if os.path.exists(model_dir):
    rmtree(model_dir)
os.makedirs(model_dir)

# data preparation
logging.info('get dataset')
properties = ['energy', 'forces']
dataset = AtomsData(os.path.join(data_dir, 'ethanol.db'),
                    required_properties=properties)
train, val, test = train_test_split(data=dataset, num_train=args.split[0],
                                    num_val=args.split[1],
                                    split_file='training/split.npz')
train_loader = AtomsLoader(train, batch_size=args.batch_size)
val_loader = AtomsLoader(val, batch_size=args.batch_size)
test_loader = AtomsLoader(test, batch_size=args.batch_size)
atomrefs = dataset.get_atomrefs(properties)
means, stddevs = train_loader.get_statistics(properties, atomrefs=atomrefs)

# model build
logging.info('build model')
representation = SchNet(n_interactions=6)
output_modules = [Atomwise(property='energy', derivative='forces',
                           mean=means['energy'], stddev=stddevs['energy'],
                           negative_dr=True)]

model = AtomisticModel(representation, output_modules)

# hooks
logging.info('build trainer')
metrics = [MeanAbsoluteError(p, p) for p in properties]
logging_hooks = [TensorboardHook(log_path=model_dir, metrics=metrics),
                 CSVHook(log_path=model_dir, metrics=metrics)]
scheduleing_hooks = [ReduceLROnPlateauHook(patience=25, window_length=3, factor=0.8)]
hooks = logging_hooks + scheduleing_hooks

# trainer
loss = loss_fn(properties)
trainer = Trainer(model_dir, model=model, hooks=hooks, loss_fn=loss,
                  optimizer=Adam(params=model.parameters(), lr=1e-4),
                  train_loader=train_loader,
                  validation_loader=val_loader)

# run training
logging.info('training')
trainer.train(device=args.device)