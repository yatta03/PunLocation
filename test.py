import time
import torch
import torch.optim as optim
import os
import sys
from models import LM_LSTM_CRF, ViterbiLoss
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import WCDataset
from inference import ViterbiDecoder
from sklearn.metrics import f1_score
from config import Config
import copy
import argparse

# fix random seed
seed = 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=seed)
else:
    torch.manual_seed(seed=seed)
np.random.seed(seed=seed)

config = Config("all", 10, 50)
# print(config)

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("cudnn version:", torch.backends.cudnn.version())

task = "all"
batch_size=10
epoch=10
fold = 0
config = Config(task, batch_size, epoch)
print(config)


fold_num = 10
f1_score = 0.0
b_true_pos = 0
b_false_pos = 0
b_true_neg = 0
b_false_neg = 0
b_loc_true_pos = 0
b_loc_false_pos = 0
b_loc_true_neg = 0
b_loc_false_neg = 0
b_loc_multi_pos = 0
global best_f1, epochs_since_improvement, checkpoint, start_epoch, word_map, char_map, tag_map
global sentences, tags, pos_mask
# load hetero
train_words, train_tags, train_pos_mask, val_words, val_tags, val_pos_mask, test_words, test_tags, test_pos_mask = [],[],[],[],[],[],[],[],[]

# load hetero
sentences1, tags1, pos_mask1 = load_sentences("datasets/subtask1-heterographic-test.xml", "datasets/subtask1-heterographic-test.gold",
                                                "datasets/subtask2-heterographic-test.xml", "datasets/subtask2-heterographic-test.gold",
                                                config.use_all_instances, isDebug=config.debug)
train_words1, train_tags1, train_pos_mask1, \
val_words1, val_tags1, val_pos_mask1,\
test_words1, test_tags1, test_pos_mask1\
    = get_n_fold_splitting(sentences1, tags1, pos_mask1, fold, fold_num)

# load homo
sentences2, tags2, pos_mask2 = load_sentences("datasets/subtask1-homographic-test.xml", "datasets/subtask1-homographic-test.gold",
                                                "datasets/subtask2-homographic-test.xml", "datasets/subtask2-homographic-test.gold",
                                                config.use_all_instances, isDebug=config.debug)
train_words2, train_tags2, train_pos_mask2, \
val_words2, val_tags2, val_pos_mask2,\
test_words2, test_tags2, test_pos_mask2\
    = get_n_fold_splitting(sentences2, tags2, pos_mask2, fold, fold_num)

# concat
sentences = sentences1 + sentences2
tags = tags1 + tags2
pos_mask = pos_mask1 + pos_mask2

train_words, train_tags, train_pos_mask = concat_shuffle(train_words1, train_words2, train_tags1, train_tags2, train_pos_mask1, train_pos_mask2)
val_words, val_tags, val_pos_mask = concat_shuffle(val_words1, val_words2, val_tags1, val_tags2, val_pos_mask1, val_pos_mask2)
test_words, test_tags, test_pos_mask = concat_shuffle(test_words1, test_words2, test_tags1, test_tags2, test_pos_mask1, test_pos_mask2)

print("\ndata len: total / hetero / homo")
print(f"total: {len(sentences)}: {len(sentences1)} + {len(sentences2)}")
print(f"train: {len(train_words)}: {len(train_words1)} + {len(train_words2)}")
print(f"val: {len(val_words)}: {len(val_words1)} + {len(val_words2)}")
print(f"test: {len(test_words)}: {len(test_words1)} + {len(test_words2)}")