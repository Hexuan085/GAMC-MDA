## Anti-MDA
Code for 'Generic and Anti-MDA Model Certification for Intellectual Property Protection in MLaaS'

## Produce a marked model for registration
This part is adapted from [Watermark-Robustness-Toolbox](https://github.com/dnn-security/Watermark-Robustness-Toolbox/tree/master). Several watermark schemes: [Content](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M), 
  [Noise](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M),
  [Unrelated](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M),[Jia](https://www.usenix.org/conference/usenixsecurity21/presentation/jia), 
  [Frontier Stitching](https://link.springer.com/article/10.1007/s00521-019-04434-z),
  [Blackmarks](https://arxiv.org/abs/1904.00344),[DeepSigns](https://dl.acm.org/doi/abs/10.1145/3297858.3304051),
  [DeepMarks](https://dl.acm.org/doi/abs/10.1145/3323873.3325042) are embeded into two neural networks: WRN and Densenet. Dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (32x32 pixels, 10 classes) is utilized for the training.

The following three main scripts provide the entire watermark related work's functionality:

- *train.py*: Pre-trains an unmarked neural network. 
- *embed.py*: Embeds a watermark into a pre-trained neural network. 
- *steal.py*: Performs a removal attack against a watermarked neural network.

### Step 1: Pre-train a Model on CIFAR-10
```shell
$ python train.py --config configs/cifar10/train_configs/resnet.yaml
```
This creates an ``outputs`` directory and saves a model file at ``outputs/cifar10/null_models/resnet/``.

### Step 2: Embed an Jia Watermark
```shell
$ python embed.py --wm_config configs/cifar10/wm_configs/Jia.yaml \
                  --filename outputs/cifar10/null_models/resnet/best.pth
```
This embeds an Jia watermark into the pre-trained model from 'Step 1' and saves (i) the watermarked model and
(ii) all data to read the watermark under ``outputs/cifar10/wm/Jia/00000_Jia/``. We utilize this script to achieve three goals: (a) produce the marked models for registration. (b) Embed the registered watermark to a pre-trained model with bad performance, which is regarded as the performance defamation attack. (c) Embed a new watermark to a registered model, which is regarded as the maliculous information defamation.

### Step 3: Attempt to Remove a Watermark
```shell
$ python steal.py --attack_config configs/cifar10/attack_configs/ftal.yaml \
                  --wm_dir outputs/cifar10/wm/adi/00000_adi/
```
This runs the Fine-Tuning (FTAL) removal attack against the watermarked model and creates a surrogate model stored under
``outputs/cifar10/attacks/ftal/``. The directory also contains human-readable debug files, such as the surrogate model's watermark and 
test accuracies. 

### Step 4: Evaluate the Watermark Accuracy
```shell
$ python wmk_acc_cal_deepsign.py --wm_config configs/cifar10/wm_configs/jia.yaml --pretrained_dir1  outputs/cifar10/wm/adi/00000_adi --pretrained_dir outputs/cifar10/poisoning_attack/00006_jia
```
we can check the accuracy of watermark embeded into '00000_adi' for model '00006_jia' where the watermark 'Jia' is embeded to the marked model '00000_adi'.

## Watermrak Registration

- *wmk_test.py*: Generate the membership and witness.
- *sign_time.py*: Generate the signature.
- *certification.py*: Generate the OR protocol.
