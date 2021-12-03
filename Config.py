# -*- coding: utf-8 -*-
import os
from random import Random, random
import torch
import time
import ml_collections
import random
import numpy as np

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 60

pretrain = False
task_name = 'MoNuSeg' # GlaS MoNuSeg
# task_name = 'GlaS'
learning_rate = 1e-3
batch_size = 4


model_name = 'LSUnetMix'
# model_name = 'LSUnetMix_pretrain'

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'



def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_layers = 1
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.embeddings_dropout_rate = 0.1
    config.attention_dropout_rate = 0.1
    config.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

