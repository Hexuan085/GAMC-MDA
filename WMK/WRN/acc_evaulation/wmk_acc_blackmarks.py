""" This script trains null models given a configuration file (see configs) """

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from shutil import copyfile

import mlconfig
import numpy as np
import torch

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training
from wrt.attacks import RemovalAttack

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets.cifar10 import cifar_classes
from wrt.utils import reserve_gpu, get_max_index

from mlconfig import instantiate
from mlconfig import load
from mlconfig import register

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--wm_dir", type=str,
                        default="outputs/imagenet/wm/jia/00000_jia",
                        help="Path to the directory with the watermarking files. "
                             "This scripts expects a 'best.pth' and one '*.yaml' file "
                             "to exist in this dir.")
    parser.add_argument('-r', "--resume", type=str,
                        default=None,
                        help="Path to checkpoint to continue the attack. ")
    parser.set_defaults(true_labels=False, help="Whether to use ground-truth labels.")
    parser.add_argument('--true_labels', action='store_true')
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")

    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, defense_filename: str = None):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    # Load defense model from a saved state, if available.
    # We allow loading the optimizer, as it only loads states that the attacker could tune themselves (E.g. learning rate)
    if defense_filename is not None:
        pretrained_data = torch.load(defense_filename)
        model.load_state_dict(pretrained_data["model"])
        try:
            optimizer.load_state_dict(pretrained_data["optimizer"])
        except:
            print("Optimizer could not be loaded. ")
            pass

        print(f"Loaded model and optimizer from '{defense_filename}'.")

    model = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)
    device = "cuda"

    # Discover the '*.yaml' config file and the 'best.pth' file.
    defense_yaml = file_with_suffix_exists(dirname=args.wm_dir, suffix=".yaml")
    pth_file = file_with_suffix_exists(dirname=args.wm_dir, suffix="checkpoint.pth")

    if not defense_yaml or not pth_file:
        raise FileNotFoundError(defense_yaml)

    defense_config = mlconfig.load(defense_yaml)
    print(defense_config)
    pth_file = '/home/xhe085/Proj/NEW_WMK_copy/Watermark-Robustness-Toolbox/best_point/00006_blackmarks/checkpoint.pth'
    model_basedir, model_filename = os.path.split(pth_file)

    pth_file_model = '/home/xhe085/Proj/NEW_WMK_copy/Watermark-Robustness-Toolbox/best_point/00006_blackmarks/checkpoint.pth'
    source_model = instantiate(defense_config.source_model)
    source_model = source_model.to(device)
    optimizer = instantiate(defense_config.optimizer,source_model.parameters())
    source_model = __load_model(source_model, optimizer,
                                image_size=defense_config.source_model.image_size,
                                num_classes=defense_config.source_model.num_classes,
                                defense_filename=pth_file_model)

    defense = instantiate(defense_config.wm_scheme, classifier=source_model, optimizer=optimizer, config=defense_config)
    x_wm, y_wm = defense.load(filename=model_filename, path=model_basedir)

    train_loader = instantiate(defense_config.dataset, train=True)
    valid_loader = instantiate(defense_config.dataset, train=False)

    source_test_acc_before_attack = evaluate_test_accuracy(source_model, valid_loader)
    print(f"Source model test acc: {source_test_acc_before_attack:.4f}")
    source_wm_acc = defense.verify(x_wm, y_wm, classifier=source_model)[0]
    print(f"Source model wm acc: {source_wm_acc:.4f}")

if __name__ == "__main__":
    main()
