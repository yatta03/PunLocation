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
        print('\npred\n', decoded)
        # print('gold\n', tmaps_sorted)
        # print('tag_map 1: ', tag_map.get(1))

        # simulated evaluation
        if torch.cuda.is_available():
            pun_tag_id = tag_map.get(1)
        else:
            pun_tag_id = tag_map.get("1")
        golds = tmaps_sorted
        preds = decoded
        result_is_pun = False
        result_pun_loc = 0
        for g, p in zip(golds, preds):
            is_pun = False
            gold = g.cpu().numpy()
            pred = p.cpu().numpy()
            # print("\nresult:")
            if pun_tag_id in gold and pun_tag_id in pred:
                # print("pun!")
                is_pun = True
                result_is_pun = True
            elif pun_tag_id in gold and pun_tag_id not in pred:
                # print("not pun")
                is_pun = True
            elif pun_tag_id not in gold and pun_tag_id not in pred:
                result_is_pun = False
                # print("not pun!")
                # true_neg += 1
            elif pun_tag_id not in gold and pun_tag_id in pred:
                # print("pun")
                result_is_pun = True
                # false_pos += 1
            # print("prediction loc: ", np.where(pred == pun_tag_id)[0])
            result_pun_loc = np.where(pred == pun_tag_id)[0]
            if is_pun:
                # loc_total += 1
                idx = np.where(gold == pun_tag_id)
                print("predict position: ", pred[idx[0]])
                if pred[idx[0]] == pun_tag_id:
                    tag_occ_count = len(np.where(pred == pun_tag_id)[0])
                    if tag_occ_count == 1:
                        print("true position")
                        # loc_true_pos += 1
                    else:
                        print("false position")
                        # loc_false_pos += 1
                        # loc_multi_pos += 1
                else:
                    if pun_tag_id not in pred:
                        print("loc false neg")
                        # loc_false_neg += 1
                    else:
                        print("loc false position")
                        # loc_false_pos += 1

    return result_is_pun, result_pun_loc

# fix random seed
seed = 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=seed)
else:
    torch.manual_seed(seed=seed)

# load the best model in each fold
# load test data and make prediction

fold = 1
task = "all"
batch_size = 10
epoch = 50
fold_num = 10
config = Config(task, batch_size, epoch)
# print(config)

# prepare for loading data
global best_f1, epochs_since_improvement, checkpoint, start_epoch, word_map, char_map, tag_map
global sentences, tags, pos_mask
train_words, train_tags, train_pos_mask, val_words, val_tags, val_pos_mask, test_words, test_tags, test_pos_mask = [],[],[],[],[],[],[],[],[]
# load data
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

word_map, char_map, tag_map, mask_map = create_maps(train_words + val_words, train_tags + val_tags, train_pos_mask+val_pos_mask,
                                                  config.min_word_freq, config.min_char_freq)
temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}

# load test data and convert to tensor
# test_inputs = create_input_tensors(test_words[:3], test_tags[:3], temp_word_map, char_map, tag_map, test_pos_mask, mask_map)
# test_loader = torch.utils.data.DataLoader(WCDataset(*test_inputs), batch_size=config.batch_size, shuffle=True,
#                                              num_workers=config.workers, pin_memory=False)

parser = argparse.ArgumentParser(description="Parse a command string into a list of lists.")
parser.add_argument(
    "--input", 
    type=str, 
    required=True, 
    help="The pun string to parse. split with white space"
)
parser.add_argument(
    "--fold", 
    type=int, 
    default=1,
    help="The fold number of the used model is trained"
)

# Parse the arguments
args = parser.parse_args()

# Parse the command string
# input = "look on the sunny side of life ."
# test_in = [['Look', 'on', 'the', 'sunny', 'side', 'of', 'life', '.']]
test_in = parse_command(args.input)
fold = args.fold

test_in_tag = [[0 for _ in test_in[0]]]
test_in_pos_mask = [[1 if i >= len(test_in[0])//2 else 0 for i, _ in enumerate(test_in[0])]]
test_inputs = create_input_tensors(test_in,test_in_tag, temp_word_map, char_map, tag_map, test_in_pos_mask, mask_map)
test_loader = torch.utils.data.DataLoader(WCDataset(*test_inputs), batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.workers, pin_memory=False)

print("test: ", test_in)
print("test_tags: ", test_in_tag)
print("test_post_mask: ", test_in_pos_mask)
# # print 3 example 
# for i in range(3):
#   print("test: ", test_words[i])
#   print("test_tags: ", test_tags[i])
#   print("test_post_mask: ", test_pos_mask[i])
  
# load from path
model_path = 'model/BEST_all_fold{0}_checkpoint_lm_lstm_crf.pth.tar'.format(fold)
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

lm_criterion = nn.CrossEntropyLoss().to(config.device)
crf_criterion = ViterbiLoss(tag_map, config).to(config.device)

vb_decoder = ViterbiDecoder(tag_map)

is_pun, pun_loc = eval(test_loader=test_loader,
                        model=model,
                        crf_criterion=crf_criterion,
                        vb_decoder=vb_decoder,
                        tag_map=tag_map, config=config)
print(f"\nresult: \n{'pun' if is_pun else 'not pun'}")
print(f"pun location: {pun_loc}, word: {[test_in[0][i] for i in pun_loc]}")