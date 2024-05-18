""" This script trains null models given a configuration file (see configs) """
import hashlib
import argparse
import json
import os
from wrt.training.utils import compute_accuracy

import mlconfig
import torch
from tqdm import tqdm
import time

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training
import matplotlib.pyplot as plt

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
        # x_wm, y_wm = state_dict['X_key'], state_dict['y_wm'] #for deepsign only
        #x_wm, y_wm = state_dict['x_wm'], state_dict['signatrue'] #for deepmarks only
    # print(x_wm['0'].shape) #for deepsign only
    return x_wm, y_wm
    # return x_wm['0'], y_wm  #for deepsign only

def visualize_key(x_wm: np.ndarray, output_dir: str = None):
    """ Visualizes the watermarking key.
    """
    for i in range(x_wm.shape[0]):
        plt.axis('off')
        plt.imshow(x_wm[i].transpose((1, 2, 0)), aspect='auto')
        # plt.subplots_adjust(hspace=0, wspace=0)
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"wm_sample{i}.png"))
            plt.show()

def hash_key(x_wm: np.ndarray, y_wm:np.ndarray, output_dir: str = None):
    """ Visualizes the watermarking key.
    """
    member = []
    mean_time = []
    for i in range(x_wm.shape[0]):
        tik = time.time()
        hash_value = hashlib.sha256(x_wm[i]).hexdigest()
        random_number = int(hash_value, 16)
        print(random_number)
        random_element = str(random_number) + str(int(y_wm[i]))
        print(random_element)
        tok = time.time()
        member.append(random_element)
        mean_time.append(tok-tik)
    
    print('mean_time', np.mean(mean_time))
        # plt.subplots_adjust(hspace=0, wspace=0)
    if output_dir is not None:
        with open(f'{output_dir}/wmk_mem.txt', 'w') as f:
            for number in member:
                # 写入数字，后面跟一个换行符
                f.write(f"{number}\n")

def hash_key_deepsign(y_wm:np.ndarray, output_dir: str = None):
    """ Visualizes the watermarking key.
    """
    member = []
    # length_y = len(y_wm[0]) #deepsign
    length_y = len(y_wm) #deepsign
    t1 = time.time()
    print('t1', t1)
    for i in range(length_y):
        random_element = str(i) + str(int(y_wm[i])) + str(int(t1))
        # random_element = str(i) + str(int(y_wm[i][0])) + str(int(t1)) #deepsign
        print(random_element)
        member.append(int(random_element))
        # plt.subplots_adjust(hspace=0, wspace=0)
    if output_dir is not None:
        with open(f'{output_dir}/wmk_mem.txt', 'w') as f:
            for number in member:
                # 写入数字，后面跟一个换行符
                f.write(f"{number}\n")
            

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

    output_dir = f'outputs/cifar10/wm_png/{defense_config.name}'
    os.makedirs(output_dir, exist_ok=False)
    print(f"===========> Creating directory '{output_dir}'")
    # hash_key_deepsign(y_wm = y_wm, output_dir=output_dir)
    tik = time.time()
    hash_key(x_wm = x_wm, y_wm = y_wm, output_dir=output_dir)
    tok = time.time()
    print('time', tok - tik)
    visualize_key(x_wm = x_wm, output_dir=output_dir)

    checkpoint = {
                'x_wm': x_wm,
                'y_wm': y_wm
            }
    torch.save(checkpoint, f'{output_dir}/wmk.pth')

    source_test_acc_before_attack = evaluate_test_accuracy(source_model, valid_loader)
    print(f"Source model test acc (before): {source_test_acc_before_attack}")

    metrics: dict = compute_metrics(defense, x_wm, y_wm)
    print("x_wm, y_wm",x_wm, y_wm)
    print("Source model test acc: {}".format(source_test_acc_before_attack))
    print("Source model wm acc: {}".format(metrics["wm_acc"]))

if __name__ == "__main__":
    main()
