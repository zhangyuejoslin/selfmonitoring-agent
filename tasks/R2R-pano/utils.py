import os
import sys
import re
import string
import json
import time
import math
import shutil
import warnings
from collections import Counter
import numpy as np
import networkx as nx
import spacy

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable

nlp1 = spacy.load("en_core_web_lg")
nlp2 = spacy.load("en_vectors_web_lg")

object_class = []
base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')
config = {
    'split_file' : 'tasks/R2R-pano/data/data/split_dictionary.txt',
    'stop_words_file': 'tasks/R2R-pano/data/data/stop_words.txt'
}
with open(config['split_file']) as f_dict:
    dictionary = f_dict.read().split('\n')
with open(config["stop_words_file"]) as f_stop_word:
    stopword = f_stop_word.read().split('\n')


def setup(opts, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Check for vocabs
    if not os.path.exists(opts.train_vocab):
        write_vocab(build_vocab(splits=['train']), opts.train_vocab)
    if not os.path.exists(opts.trainval_vocab):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), opts.trainval_vocab)


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def post_processing_sentence(sentence_list):
    def func(sl):
        sl = sl.strip()
        sl = sl.strip(".")
        for sw in stopword:
            if sl.endswith(" %s"%sw):
                sl = sl[:-(len(sw)+1)]
            elif sl.startswith("%s "%sw):
                sl = sl[len(sw)+1:]
        return sl
    sentence_list = list(map(func, sentence_list))
    new_sentence_list = []
    for sent in sentence_list:
        if sent !='':
            new_sentence_list.append(sent)
    return new_sentence_list

def get_noun_chunks(each_configuration):
    doc = nlp1(each_configuration)
    for chunk in doc.noun_chunks:
        tokens = nlp2(chunk.root.text).vector
        return tokens
                
def get_configurations(sentence):
    sentence_list = []
    sentence = sentence.lower().strip()
    sentence_list = sentence.split('.')
    for each_word in dictionary:
        for sl in sentence_list:
            if each_word in sl:
                index = sentence_list.index(sl)
                sentence_list.remove(sl)
                temp = sl.split(each_word)
                temp = [tt if id== 0 else each_word+tt for id,tt in enumerate(temp)]
                for tt in temp:
                    if tt:
                        sentence_list.insert(index,tt)
                        index +=1
    sentence_list = post_processing_sentence(list(filter(None,sentence_list)))
    return sentence_list


def load_datasets(splits, opts=None):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test', 'train_val_seen', 'synthetic']
        if split == 'synthetic':
            with open('tasks/R2R-pano/data/R2R_literal_speaker_data_augmentation_paths.json') as f:
                data += json.load(f)
        else:
            with open('tasks/R2R-pano/data/data/R2R_%s_small.json' % split) as f:
                data += json.load(f)

    return data


class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, remove_punctuation=False, reversed=True, vocab=None, encoding_length=20):
        self.remove_punctuation = remove_punctuation
        self.reversed = reversed
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.word_to_index = {}
        # self.client = CoreNLPClient(default_annotators=['ssplit', 'tokenize', 'pos'])
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []

        splited = self.split_sentence(sentence)
        if self.reversed:
            splited = splited[::-1]

        if self.remove_punctuation:
            splited = [word for word in splited if word not in string.punctuation]

        for word in splited:  # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])

        encoding.append(self.word_to_index['<EOS>'])
        encoding.insert(0, self.word_to_index['<START>'])

        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:self.encoding_length])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        if self.reversed:
            sentence = sentence[::-1]
        return " ".join(sentence)


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    """ Build a vocab, starting with base vocab containing a few useful tokens. """
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word, num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab), path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def set_tb_logger(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints/', epoch_num=0, name='checkpoint'):
    os.makedirs(checkpoint_dir + name, exist_ok=True)
    filename = checkpoint_dir + name + str(epoch_num) + '.pth'
    #filename = checkpoint_dir + name + "" + str(epoch_num) + '.pth'
   
    torch.save(state, filename)
    if is_best:
        best_filename = checkpoint_dir + name + str(epoch_num)+'_model_best.pth.tar'
        shutil.copyfile(filename, best_filename)


def is_experiment():
    """
    A small function for developing on MacOS. When developing, the code will not load the full dataset
    """
    if sys.platform != 'darwin':
        return True
    else:
        return False


def resume_training(opts, model, encoder, optimizer):
    if opts.resume == 'latest':
        file_extention = '.pth'
    elif opts.resume == 'best':
        file_extention = '_model_best.pth.tar'
    else:
        raise ValueError('Unknown resume option: {}'.format(opts.resume))
    #opts.resume = opts.checkpoint_dir + "experiments_20200611-031353/" + "285" + file_extention
    # main:experiments_20200622-044027/" + "105"
    # 80: experiments_20200620-212112/" + "105" 
    # soft atten: experiments_20200630-143113.txt
    opts.resume = opts.checkpoint_dir + "experiments_20200630-143113/" + "105" + file_extention
    #opts.resume = opts.checkpoint_dir + opts.exp_name + file_extention
    if os.path.isfile(opts.resume):
        if is_experiment():
            checkpoint = torch.load(opts.resume)
        else:
            checkpoint = torch.load(opts.resume, map_location=lambda storage, loc: storage)

        opts.start_epoch = checkpoint['epoch']
        try:
            opts.max_episode_len = checkpoint['max_episode_len']
        except:
            pass
        model.load_state_dict(checkpoint['state_dict'])
        if encoder is not None:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            best = checkpoint['best_success_rate']
        except:
            best = checkpoint['best_loss']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opts.resume, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(opts.resume))

    return model, encoder, optimizer, best

def pad_tensor(tensor, length):
    """Pad a tensor, given by the max length"""
    if tensor.size(0) == length:
        return tensor
    return torch.cat([tensor, tensor.new(length - tensor.size(0),
                                  *tensor.size()[1:]).zero_()])

def find_length(list_tensors):
    """find the length of list of tensors"""
    if type(list_tensors[0]) is np.ndarray:
        length = [x.shape[0] for x in list_tensors]
    else:
        length = [x.size(0) for x in list_tensors]
    return length

def pad_list_tensors(list_tensor, max_length=None):
    """Pad a list of tensors and return a list of tensors"""
    tensor_length = find_length(list_tensor)

    if max_length is None:
        max_length = max(tensor_length)

    list_padded_tensor = []
    for tensor in list_tensor:
        if tensor.size(0) != max_length:
            tensor = pad_tensor(tensor, max_length)
        list_padded_tensor.append(tensor)

    return torch.stack(list_padded_tensor), tensor_length

