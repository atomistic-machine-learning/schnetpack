#!/usr/bin/env python
import argparse
import logging
import os
import sys
from shutil import copyfile, rmtree

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
from schnetpack.datasets import OrganicMaterialsDatabase
from schnetpack.utils import to_json, read_from_json, compute_params

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()

    ## command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument('--cuda', help='Set flag to use GPU(s)', action='store_true')
    cmd_parser.add_argument('--parallel',
                            help='Run data-parallel on all available GPUs (specify with environment variable'
                                 + ' CUDA_VISIBLE_DEVICES)', action='store_true')
    cmd_parser.add_argument('--batch_size', type=int,
                            help='Mini-batch size for training and prediction (default: %(default)s)',
                            default=32)
    ## training
    train_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    train_parser.add_argument('datapath', help='Path / destination of OMDB dataset directory')
    train_parser.add_argument('modelpath', help='Destination for models and logs')
    train_parser.add_argument('--property', type=str,
                              help='Organic Materials Database property to be predicted (default: %(default)s)',
                              default="band_gap", choices=OrganicMaterialsDatabase.properties)
    train_parser.add_argument('--seed', type=int, default=None, help='Set random seed for torch and numpy.')
    train_parser.add_argument('--overwrite', help='Remove previous model directory.', action='store_true')

    train_parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs (default: %(default)s)',
                              default=5000)
    train_parser.add_argument('--lr', type=float, help='Initial learning rate (default: %(default)s)',
                              default=1e-3)
    train_parser.add_argument('--lr_patience', type=int,
                              help='Epochs without improvement before reducing the learning rate (default: %(default)s)',
                              default=10)
    train_parser.add_argument('--lr_decay', type=float, help='Learning rate decay (default: %(default)s)',
                              default=0.8)
    train_parser.add_argument('--lr_min', type=float, help='Minimal learning rate (default: %(default)s)',
                              default=1e-4)

    train_parser.add_argument('--logger', help='Choose logger for training process (default: %(default)s)',
                              choices=['csv', 'tensorboard'], default='csv')
    train_parser.add_argument('--log_every_n_epochs', type=int,
                              help='Log metrics every given number of epochs (default: %(default)s)',
                              default=1)

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path of OMDB dataset directory')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--split', help='Evaluate trained model on given split',
                             choices=['train', 'validation', 'test'], default=['test'], nargs='+')

    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--aggregation_mode', type=str, default='avg', choices=['sum', 'avg'],
                              help=' (default: %(default)s)')

    #######  SchNet  #######
    schnet_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    schnet_parser.add_argument('--features', type=int, help='Size of atom-wise representation (default: %(default)s)',
                               default=64)
    schnet_parser.add_argument('--interactions', type=int, help='Number of interaction blocks (default: %(default)s)',
                               default=3)
    schnet_parser.add_argument('--cutoff', type=float, default=5.,
                               help='Cutoff radius of local environment (default: %(default)s)')
    schnet_parser.add_argument('--num_gaussians', type=int, default=25,
                               help='Number of Gaussians to expand distances (default: %(default)s)')

    ## setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode', help='Command-specific arguments')
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser('train', help='Training help')
    subparser_eval = cmd_subparsers.add_parser('eval', help='Training help')

    train_subparsers = subparser_train.add_subparsers(dest='model', help='Model-specific arguments')
    train_subparsers.required = True

    subparser_export = cmd_subparsers.add_parser('export', help='Export help')
    subparser_export.add_argument('modelpath', help='Path of stored model')
    subparser_export.add_argument('destpath', help='Destination path for exported model')

    train_subparsers.add_parser('schnet', help='SchNet help', parents=[train_parser, schnet_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model', help='Model-specific arguments')
    eval_subparsers.add_parser('schnet', help='SchNet help', parents=[eval_parser, schnet_parser])

    return main_parser


def train(args, model, train_loader, val_loader, device):
    # setup hook and logging
    hooks = [
        spk.train.MaxEpochHook(args.max_epochs)
    ]

    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    schedule = spk.train.ReduceLROnPlateauHook(optimizer, patience=args.lr_patience, factor=args.lr_decay,
                                               min_lr=args.lr_min,
                                               window_length=1, stop_after_min=True)
    hooks.append(schedule)

    if args.logger == 'csv':
        logger = spk.train.CSVHook(os.path.join(args.modelpath, 'log'),
                                   [spk.metrics.MeanAbsoluteError(args.property, "y"),
                                    spk.metrics.RootMeanSquaredError(args.property, "y")],
                                   every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        logger = spk.train.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                           [spk.metrics.MeanAbsoluteError(args.property, "y"),
                                            spk.metrics.RootMeanSquaredError(args.property, "y")],
                                           every_n_epochs=args.log_every_n_epochs)
        hooks.append(logger)

    # setup loss function
    def loss(batch, result):
        diff = batch[args.property] - result["y"]
        diff = diff ** 2
        err_sq = torch.mean(diff)
        return err_sq

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks)
    trainer.train(device)


def evaluate(args, model, property, train_loader, val_loader, test_loader, device):
    header = ['Subset', property + ' MAE', property + ' RMSE']

    metrics = [spk.metrics.MeanAbsoluteError(property, "y"),
               spk.metrics.RootMeanSquaredError(property, "y")
               ]

    results = []
    if 'train' in args.split:
        results.append(['training'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, train_loader, device)])

    if 'validation' in args.split:
        results.append(['validation'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, val_loader, device)])

    if 'test' in args.split:
        results.append(['test'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, test_loader, device)])

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results, header=header, fmt='%s', delimiter=',')


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [
        metric.aggregate() for metric in metrics
    ]
    return results


def get_model(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    if args.model == 'schnet':
        representation = spk.representation.SchNet(args.features, args.features, args.interactions,
                                                   args.cutoff, args.num_gaussians, normalize_filter=True)
        atomwise_output = spk.atomistic.Atomwise(args.features, aggregation_mode=args.aggregation_mode,
                                                 mean=mean, stddev=stddev,
                                                 atomref=atomref, train_embeddings=True)
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    if parallelize:
        model = nn.DataParallel(model)

    logging.info("The model you built has: {} parameters".format(compute_params(model)))

    return model


def export_model(args):
    jsonpath = os.path.join(args.modelpath, 'args.json')
    train_args = read_from_json(jsonpath)
    model = get_model(train_args, atomref=np.zeros((100, 1)))
    model.load_state_dict(
        torch.load(os.path.join(args.modelpath, 'best_model')))

    torch.save(model, args.destpath)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == 'export':
        export_model(args)
        sys.exit(0)

    device = torch.device("cuda" if args.cuda else "cpu")
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, 'args.json')

    if args.mode == 'train':
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info('existing model will be overwritten...')
            rmtree(args.modelpath)

        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        to_json(jsonpath, argparse_dict)

        spk.utils.set_random_seed(args.seed)
        train_args = args
    else:
        train_args = read_from_json(jsonpath)

    # will download OMDB dataset if necessary
    omdb = spk.datasets.OrganicMaterialsDatabase(args.datapath, args.cutoff, download=True,
                                       properties=[train_args.property])

    # splits the dataset in test, val, train sets
    split_path = os.path.join(args.modelpath, 'split.npz')
    idx = list(range(12500))
    np.savez(split_path, train_idx=idx[:9000], val_idx=idx[9000:10000], test_idx=idx[-2500:])

    data_train, data_val, data_test = omdb.create_splits(split_file=split_path)

    train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                        num_workers=4, pin_memory=True)
    val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    if args.mode == 'train':
        mean, stddev = train_loader.get_statistics(train_args.property, True)
        logging.info('Training set statistics: mean=%.3f, stddev=%.3f' % (mean.numpy(), stddev.numpy()))
    else:
        mean, stddev = None, None

    # Construct the model.
    model = get_model(train_args, atomref=None, mean=mean, stddev=stddev, train_loader=train_loader,
                      parallelize=args.parallel).to(device)

    if args.mode == 'eval':
        model = torch.load(os.path.join(args.modelpath, 'best_model'))
    if args.mode == 'train':
        logging.info("Training...")
        train(args, model, train_loader, val_loader, device)
    elif args.mode == 'eval':
        test_loader = spk.data.AtomsLoader(data_test, batch_size=args.batch_size,
                                           num_workers=2, pin_memory=True)
        evaluate(args, model, train_args.property, train_loader, val_loader, test_loader, device)
    else:
        print('Unknown mode:', args.mode)
