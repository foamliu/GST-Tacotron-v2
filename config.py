import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

num_train = 120098
num_dev = 14326
num_test = 7176

vocab = 'abcdefghijklmnopqrstuvwxyz12345 '
vocab_size = len(vocab)
idx_to_char = {i: vocab[i] for i in range(0, len(vocab))}
char_to_idx = {vocab[i]: i for i in range(0, len(vocab))}

unk_id = 0

aishell_folder = 'data/data_aishell'
wav_folder = os.path.join(aishell_folder, 'wav')
tran_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
data_file = 'data/data_aishell.pkl'

################################
# Experiment Parameters        #
################################
epochs = 500
iters_per_checkpoint = 1000
seed = 1234
dynamic_loss_scaling = True
fp16_run = False
distributed_run = False

################################
# Data Parameters             #
################################
load_mel_from_disk = False

################################
# Audio Parameters             #
################################
max_wav_value = 32768.0
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Model Parameters             #
################################
n_symbols = vocab_size
symbols_embedding_dim = 512

# Reference encoder
ref_enc_filters = [32, 32, 64, 64, 128, 128]
ref_wav = 'ref_wav/xiaofeng.wav'

# Style token layer
token_num = 1024
token_emb_size = 512
num_heads = 8

# Encoder parameters
encoder_kernel_size = 5
encoder_n_convolutions = 3
encoder_embedding_dim = 512

# Decoder parameters
n_frames_per_step = 1  # currently only 1 is supported
decoder_rnn_dim = 1024
prenet_dim = 256
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# Attention parameters
attention_rnn_dim = 1024
attention_dim = 128

# Location Layer parameters
attention_location_n_filters = 32
attention_location_kernel_size = 31

# Mel-post processing network parameters
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

################################
# Optimization Hyperparameters #
################################
learning_rate = 1e-3
weight_decay = 1e-6
batch_size = 64
mask_padding = True  # set model's padded outputs to padded values
