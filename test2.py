import time
import torch
import torch.optim as optim
import os
import sys
from models import LM_LSTM_CRF, ViterbiLoss
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datasets import WCDataset
from inference import ViterbiDecoder
from sklearn.metrics import f1_score
from config import Config
import copy
import argparse

# ignore the torch load warming, maybe resolve it later
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_command(input_string):
    # Split the string into words
    tokens = input_string.split()
    return [tokens]

def eval(test_loader, model, crf_criterion, vb_decoder, tag_map, config):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()

    batch_time = AverageMeter()
    vb_losses = AverageMeter()
    f1s = AverageMeter()
    b_true_pos = 0
    b_false_pos = 0
    b_true_neg = 0
    b_false_neg = 0
    b_loc_true_pos = 0
    b_loc_false_pos = 0
    b_loc_true_neg = 0
    b_loc_false_neg = 0
    b_loc_multi_pos = 0

    # pun classification

    start = time.time()
    instances_len = 0
    instances_len = 0
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, pos_mask) in enumerate(
            test_loader):

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(config.device)
        cmaps_f = cmaps_f[:, :max_char_len].to(config.device)
        cmaps_b = cmaps_b[:, :max_char_len].to(config.device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(config.device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(config.device)
        tmaps = tmaps[:, :max_word_len].to(config.device)
        pos_mask = pos_mask[:, :max_word_len].to(config.device)
        wmap_lengths = wmap_lengths.to(config.device)
        cmap_lengths = cmap_lengths.to(config.device)

        # Forward prop.
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                    cmaps_b,
                                                                                    cmarkers_f,
                                                                                    cmarkers_b,
                                                                                    wmaps,
                                                                                    tmaps,
                                                                                    wmap_lengths,
                                                                                    cmap_lengths,
                                                                                    pos_mask)

        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        # print('gold\n', tmaps_sorted)
        # print('tag_map 1: ', tag_map.get(1))

        f1, _, true_pos, false_pos, true_neg, \
        false_neg, loc_true_pos, loc_false_pos, \
        loc_true_neg, loc_false_neg, loc_multi_pos = calculate_f1_score(tmaps_sorted, decoded, tag_map)

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum((wmap_lengths_sorted - 1).tolist()))
        batch_time.update(time.time() - start)
        b_true_pos += true_pos
        b_false_pos += false_pos
        b_true_neg += true_neg
        b_false_neg += false_neg
        b_loc_true_pos += loc_true_pos
        b_loc_false_pos += loc_false_pos
        b_loc_true_neg += loc_true_neg
        b_loc_false_neg += loc_false_neg
        b_loc_multi_pos += loc_multi_pos
        start = time.time()
    print(
    '\n * LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}\n'.format(vb_loss=vb_losses,
                                                                          f1=f1s))
    return b_true_pos, b_false_pos, b_true_neg, b_false_neg, b_loc_true_pos, b_loc_false_pos, b_loc_true_neg, b_loc_false_neg, b_loc_multi_pos

# fix random seed
seed = 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=seed)
else:
    torch.manual_seed(seed=seed)

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

task = "homo"
batch_size = 10
epoch = 50
config = Config(task, batch_size, epoch)
# print(config)

for f in range(fold_num):
  print('Working on {0} fold'.format(f))

  # prepare for loading data
  global best_f1, epochs_since_improvement, checkpoint, start_epoch, word_map, char_map, tag_map
  global sentences, tags, pos_mask
  train_words, train_tags, train_pos_mask, val_words, val_tags, val_pos_mask, test_words, test_tags, test_pos_mask = [],[],[],[],[],[],[],[],[]
  # load data
  # load hetero
  if task == "hetero":
    sentences, tags, pos_mask = load_sentences("datasets/subtask1-heterographic-test.xml", "datasets/subtask1-heterographic-test.gold",
                                                    "datasets/subtask2-heterographic-test.xml", "datasets/subtask2-heterographic-test.gold",
                                                    config.use_all_instances, isDebug=config.debug)
    train_words, train_tags, train_pos_mask, \
    val_words, val_tags, val_pos_mask,\
    test_words, test_tags, test_pos_mask\
        = get_n_fold_splitting(sentences, tags, pos_mask, f, fold_num)
  elif task == "homo":
    # load homo
    sentences, tags, pos_mask = load_sentences("datasets/subtask1-homographic-test.xml", "datasets/subtask1-homographic-test.gold",
                                                    "datasets/subtask2-homographic-test.xml", "datasets/subtask2-homographic-test.gold",
                                                    config.use_all_instances, isDebug=config.debug)
    train_words, train_tags, train_pos_mask, \
    val_words, val_tags, val_pos_mask,\
    test_words, test_tags, test_pos_mask\
        = get_n_fold_splitting(sentences, tags, pos_mask, f, fold_num)
  else:
    print("invalid task")

  word_map, char_map, tag_map, mask_map = create_maps(train_words + val_words, train_tags + val_tags, train_pos_mask+val_pos_mask,
                                                    config.min_word_freq, config.min_char_freq)
  temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}

  
  # load from path
  model_path = 'model/BEST_all_fold{0}_checkpoint_lm_lstm_crf.pth.tar'.format(f)
  checkpoint = torch.load(model_path, map_location=config.device)

  # parse
  model = checkpoint['model']
  optimizer = checkpoint['optimizer']
  word_map = checkpoint['word_map']
  lm_vocab_size = checkpoint['lm_vocab_size']
  tag_map = checkpoint['tag_map']
  char_map = checkpoint['char_map']
  start_epoch = checkpoint['epoch'] + 1
  best_f1 = checkpoint['f1']

  # load test data and convert to tensor
  test_inputs = create_input_tensors(test_words, test_tags, word_map, char_map, tag_map, test_pos_mask, mask_map)
  test_loader = torch.utils.data.DataLoader(WCDataset(*test_inputs), batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=False)

  lm_criterion = nn.CrossEntropyLoss().to(config.device)
  crf_criterion = ViterbiLoss(tag_map, config).to(config.device)

  vb_decoder = ViterbiDecoder(tag_map)

  true_pos, false_pos, true_neg, false_neg, \
  loc_true_pos, loc_false_pos, loc_true_neg, \
  loc_false_neg, loc_multi_pos = eval(test_loader=test_loader,
                          model=model,
                          crf_criterion=crf_criterion,
                          vb_decoder=vb_decoder,
                          tag_map=tag_map, config=config)
  b_true_pos += true_pos
  b_false_pos += false_pos
  b_true_neg += true_neg
  b_false_neg += false_neg
  b_loc_true_pos += loc_true_pos
  b_loc_false_pos += loc_false_pos
  b_loc_true_neg += loc_true_neg
  b_loc_false_neg += loc_false_neg
  b_loc_multi_pos += loc_multi_pos

# print(b_false_pos, b_true_neg, b_false_neg, b_loc_true_pos, b_true_neg, b_false_neg, b_loc_multi_pos)
total = b_true_pos + b_false_pos + b_true_neg + b_false_neg
c_precision = b_true_pos * 1.0 / (b_true_pos + b_false_pos) * 100 if (b_true_pos + b_false_pos) > 0 else 0.0
c_recall = b_true_pos * 1.0 / (b_true_pos + b_false_neg) * 100 if (b_true_pos + b_false_neg) > 0 else 0.0
c_f1 = (c_precision * c_recall * 2) / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0
c_accuracy = (b_true_pos + b_true_neg) * 1.0 / total * 100 if total > 0 else 0.0

loc_total = b_loc_true_pos + b_loc_false_pos + b_loc_true_neg + b_loc_false_neg
l_precision = b_loc_true_pos * 1.0 / (b_loc_true_pos + b_loc_false_pos) * 100 if (b_loc_true_pos + b_loc_false_pos) > 0 else 0.0
l_recall = b_loc_true_pos * 1.0 / loc_total * 100 if loc_total > 0 else 0.0
l_f1 = l_precision * l_recall * 2.0 / (l_recall + l_precision) if (l_recall + l_precision) > 0 else 0.0
l_accuracy = b_loc_true_pos * 1.0 / loc_total * 100 if loc_total > 0 else 0.0


print('CV classification [{0}] instances\n'
      'Precision {prec:.3f}\t'
      'Recall {rec:.3f}\t'
      'F1 {f1:.3f}\t'
      'Acc {acc:.3f}\n'
      'CV location [{1}] pun instances\n'
      'Precision {l_prev:.3f}\t'
      'Recall {l_rec:.3f}\t'
      'F1 {l_f1:.3f}\t'
      'Acc {l_acc:.3f}\n'.format(total, loc_total, prec=c_precision, rec=c_recall, f1=c_f1, acc=c_accuracy,
                                  l_prev=l_precision,
                                  l_rec=l_recall, l_f1=l_f1, l_acc=l_accuracy))

print('Multi position prediction: {0}'.format(b_loc_multi_pos))