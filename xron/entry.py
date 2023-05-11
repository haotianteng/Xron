# Copyright 2017 The Xron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Aug 14 18:38:18 2017
import argparse
import sys
import logging
from os import path
import xron
from xron import xron_eval
from xron import xron_train_supervised
from xron import xron_init
from xron.utils import prepare_chunk
from xron.nrhmm import hmm_relabel

def check_init(init_function):
    def decorator(function):
        def wrapper(*args, **kwargs):
            init_function()
            result = function(*args, **kwargs)
            return result
        return wrapper
    return decorator

def check_model():
    model_path = path.join(xron.__path__[0], 'models')
    if not path.exists(model_path):
        logging.error('Models not found. Please run "xron init" first.')
        sys.exit(1)

@check_init(check_model)
def evaluation(args):
    xron_eval.post_args(args)
    xron_eval.main(args)

@check_init(check_model)
def export(args):
    prepare_chunk.post_args(args)
    prepare_chunk.extract(args)

@check_init(check_model)
def train(args):
    xron_train_supervised.post_args(args)
    xron_train_supervised.main(args)

@check_init(check_model)
def relabel(args):
    hmm_relabel.post_args(args)
    hmm_relabel.main(args)

def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog='xron', description='A deep neural network basecaller that achieve methylation.')
    parser.add_argument('-v','--version',action='version',version='Xron version '+xron.__version__,help="Print out the version.")
    subparsers = parser.add_subparsers(title='sub command', help='sub command help')

    # parser for 'init' command
    parser_init = subparsers.add_parser('init', description='Initialize xron package, need to run this when first time runnning xron',
                                        help='Initialize xron package, need to run this when first time runnning xron')
    parser_init.set_defaults(func=xron_init.get_models)

    # parser for 'call' command
    parser_call = subparsers.add_parser('call', description='Perform basecalling', help='Perform basecalling.')
    xron_eval.add_arguments(parser_call)
    parser_call.set_defaults(func=evaluation)
    
    # parser for 'extract' command
    parser_export = subparsers.add_parser('prepare', description='Prepare the training dataset by aligning it to the reference genome, it is an equivalent command to resquiggle when --extract_seq flag is set.',
                                          help='Realign the sequence to the reference genome and extract the signal chunk and label for training.')
    prepare_chunk.add_arguments(parser_export)
    parser_export.set_defaults(func=export)

    # parser for 'relabel' command
    parser_relabel = subparsers.add_parser('relabel', description='Relabel the training dataset using the pretraiend NHMM model.',
                                             help='Relabel the training dataset using the pretraiend NHMM model.')
    hmm_relabel.add_arguments(parser_relabel)
    parser_relabel.set_defaults(func=relabel)
    
    # parser for 'train' command
    parser_train = subparsers.add_parser('train', description='Training a model in several ways: embedding, supervised and reinforce', help='Train a model.')
    xron_train_supervised.add_arguments(parser_train)
    parser_train.set_defaults(func=train)

    args = parser.parse_args(arguments)
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #print(sys.argv[1:])
    main()