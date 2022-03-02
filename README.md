This is an official GitHub repository for the paper [Lee D, Aune E., Langet N. and Eidsvik J., 2022, "Ensemble and self-supervised learning for improved classification of seismic signals from the  ̊Aknes rockslope"]. <br>
The paper is implemented in PyTorch in this repository.

# Quick Start

:rocket: Run the self-supervised learning (SSL) of VNIbCReg: <br>
* `python sslearn.py --config_ssl configs/ssl.yaml`

:rocket: Run i) training from scratch, or ii) fine-tuning, or iii) linear evaluation: <br>
* `python finetune.py --config_ft configs/finetune.yaml`

:rocket: Run the implementation of the relevant previous study [1] as described in our paper [2].  <br>
[1] (Langet et al., 2022, "Automated classification of seismic signals recorded on the Åknes rockslope, Western Norway, using a Convolutional Neural Network") <br> 
[2] (Lee et al., 2022, "Ensemble and self-supervised learning for improved
classification of seismic signals from the  ̊Aknes rockslope") <br>
* `python original_implementation/train.py --configs original_implementation/config.yaml` <br>

Note:
* `configs/ssl.yaml` is set to run VNIbCReg by default.
* `configs/finetune.yaml` is set to run training a model (ResNet34) from scratch on 80% of the dataset.
* `original_implementation/config.yaml` is set to run training a model (ResNet34) from scratch on 80% of the dataset following the implementation of the relevant previous study [1].
* All configuration files can be edited to suit your experimental purpose.


# Configuration Files
Basic annotations are already available in the configuration files. Here, details of some parameters that might not be clear are explained. 

### ssl.yaml
- `model_params`
  - `in_channels`: input channel size
  - `out_size_enc`: output channel size from an encoder
  - `out_size_enc`: output channel size from an encoder
  - `proj_hid`: hidden size of the projector in VIbCReg
  - `proj_out`: output size of the projector in VIbCReg
  - `backbone_type`: a type of backbone. Available backbone types are `ResNet18Encoder`, `ResNet34Encoder`, `ResNet50Encoder`, and `ResNet152Encoder`. (for `ResNet50Encoder`, and `ResNet152Encoder`, `out_size_enc` needs to 2048.)

- `exp_params`
  - `LR`: learning rate
  - `model_save_ep_period`: a period for saving a model (epoch)

- `trainer_params`
  - `gpus`: indices for gpus to be used. 

- `dataset`
  - `num_workers`: `num_workers` in `torch.utils.data.DataLoader`.
  - `return_single_spectrogram_train`: use of the ensemble prediction during training
  - `return_single_spectrogram_train`: use of the ensemble prediction during testing


### finetune.yaml

Parameters that are described above are not specified.

- `load_encoder`
  - `ckpt_fname`: `none` for training from scratch. `checkpoints/some_saved_model.ckpt` for loading a pretrained encoder.

- `exp_params`
  - `freeze_encoders`: freeze the encoder if `True` to conduct _linear evaluation_ 
  - `freeze_bn_stat_train`: if `True`, `encoder.eval()` is set during training to use the averaged statistics of BatchNorm. This is useful when finetuning with a very-small dataset. 

- `dataset`
  - `train_data_ratio`: can be adjusted to 0.05 or 0.1 for the fine-tuning evaluation in a small-dataset regime.
  

### original_implementation/config.yaml

- `backbone_with_clf_type`: available backbone types are: `AlexNet`, `ResNet18`, and `ResNet34`


