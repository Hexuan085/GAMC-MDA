""" This script trains null models given a configuration file (see configs) """

import argparse
import json
import os
from wrt.training.utils import compute_accuracy

import mlconfig
import torch
from tqdm import tqdm

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.defenses import Watermark
from wrt.utils import reserve_gpu, get_max_index

from mlconfig import instantiate
from mlconfig import load
from mlconfig import register

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wm_config', type=str, default='configs/cifar10/wm_configs/dawn1.yaml',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--pretrained_dir", default="outputs/cifar10/wm/pretrained/resnet/00000_pretrained")
    parser.add_argument("--pretrained_dir1", default="outputs/cifar10/wm/pretrained/resnet/00000_pretrained")
    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, pretrained_dir: str = None,
                 best=False, load_optimizer=False):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    if pretrained_dir:
        print(f"Loading source model from '{pretrained_dir}'.")
        for file in os.listdir(pretrained_dir):
            if best:
                if file.endswith(".pth"):
                    model.load_state_dict(torch.load(os.path.join(pretrained_dir, file))["model"])
                    print(f"Loaded model '{file}'")
            elif file.endswith(".model"):
                model.load_state_dict(torch.load(os.path.join(pretrained_dir, file)))
                print(f"Loaded model '{file}'")

            if load_optimizer and file.endswith(".optimizer"):
                optimizer.load_state_dict(torch.load(os.path.join(pretrained_dir, file)))
                print(f"Loaded optimizer '{file}'.")

    model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model

def compute_metrics(defense_instance, x_wm, y_wm):
    
    source_model = defense_instance.get_classifier()
    wm_acc = defense_instance.verify(x_wm, y_wm)[0]

    return {
        "wm_acc": float(wm_acc),
    }

def __load_wmk(pretrained_dir: str = None,
                 filename: str = 'checkpoint.pth'):
    """ Loads a (pretrained) source model from a directory and wraps it into a PyTorch classifier.
    """
    if pretrained_dir:
        assert filename.endswith(".pth"), "Only '*.pth' are allowed for pretrained models"
        print(f"Loading a pretrained source model from '{pretrained_dir}'.")
        state_dict = torch.load(os.path.join(pretrained_dir, filename))
        x_wm, y_wm = state_dict['x_wm'], state_dict['y_wm']
    print(x_wm.shape)
    return x_wm, y_wm

def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)

    defense_config = mlconfig.load(args.wm_config)
    print(defense_config)

    source_model: torch.nn.Sequential = instantiate(defense_config.source_model)
    optimizer = instantiate(defense_config.optimizer, source_model.parameters())

    source_model: PyTorchClassifier = __load_model(source_model,
                                                   optimizer,
                                                   best=True,
                                                   load_optimizer=True,
                                                   image_size=defense_config.source_model.image_size,
                                                   num_classes=defense_config.source_model.num_classes,
                                                   pretrained_dir=args.pretrained_dir)
    
    train_loader = instantiate(defense_config.dataset, train=True)
    valid_loader = instantiate(defense_config.dataset, train=False)

    x_wm, y_wm = __load_wmk(pretrained_dir=args.pretrained_dir1)
    print(y_wm)
    # x_wm = train_loader.normalize(x_wm)  # only for unrelated, (noise, content不包括) (zhang)
    defense: Watermark = instantiate(defense_config.wm_scheme, source_model, config=defense_config)

    # x_wm = x_wm.detach().clone().cpu().numpy() #for deepsign only

    source_test_acc_before_attack = evaluate_test_accuracy(source_model, valid_loader)
    print(f"Source model test acc (before): {source_test_acc_before_attack}")

    metrics: dict = compute_metrics(defense, x_wm, y_wm)
    print("x_wm, y_wm",x_wm, y_wm)
    print("Source model test acc: {}".format(source_test_acc_before_attack))
    print("Source model wm acc: {}".format(metrics["wm_acc"]))

if __name__ == "__main__":
    main()
