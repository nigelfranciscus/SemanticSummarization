from __future__ import print_function
import numpy as np
import os

import cntk as C


# Define a test environment
def isTest():
    return ('TEST_DEVICE' in os.environ)


# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

# Test for CNTK version
if not C.__version__ == "2.0":
    raise Exception("this notebook was designed to work with 2.0. Current Version: " + C.__version__)

import requests


def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)


MODEL_DIR = "."
DATA_DIR = os.path.join('..', 'Examples', 'SequenceToSequence', 'CMUDict', 'Data')
# If above directory does not exist, just use current.
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

dataPath = {
    'validation': 'tiny.ctf',
    'training': 'cmudict-0.7b.train-dev-20-21.ctf',
    'testing': 'cmudict-0.7b.test.ctf',
    'vocab_file': 'cmudict-0.7b.mapping',
}

for k in sorted(dataPath.keys()):
    path = os.path.join(DATA_DIR, dataPath[k])
    if os.path.exists(path):
        print("Reusing locally cached:", path)
    else:
        print("Starting download:", dataPath[k])
        url = "https://github.com/Microsoft/CNTK/blob/v2.0/Examples/SequenceToSequence/CMUDict/Data/%s?raw=true" % \
              dataPath[k]
        download(url, path)
        print("Download completed")
    dataPath[k] = path


# Helper function to load the model vocabulary file
def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
    vocab = [w.strip() for w in open(path).readlines()]
    i2w = {i: w for i, w in enumerate(vocab)}
    w2i = {w: i for i, w in enumerate(vocab)}

    return (vocab, i2w, w2i)


# Read vocabulary data and generate their corresponding indices
vocab, i2w, w2i = get_vocab(dataPath['vocab_file'])


def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        labels=StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
    )), randomize=is_training, max_sweeps=INFINITELY_REPEAT if is_training else 1)


def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features = C.io.StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        labels   = C.io.StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
    )), randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Train data reader
train_reader = create_reader(dataPath['training'], True)

# Validation data reader
valid_reader = create_reader(dataPath['validation'], True)

hidden_dim = 512
num_layers = 2
attention_dim = 128
attention_span = 20
attention_axis = -3
use_attention = True
use_embedding = True
embedding_dim = 200
vocab = ([w.strip() for w in open(dataPath['vocab_file']).readlines()]) # all lines of vocab_file in a list
length_increase = 1.5

sentence_start =C.Constant(np.array([w=='<s>' for w in vocab], dtype=np.float32))
sentence_end_index = vocab.index('</s>')

