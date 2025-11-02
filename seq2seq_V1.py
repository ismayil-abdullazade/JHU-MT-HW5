#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Homework 5: Complete Implementation
This includes:
- Part 1: Batching and PyTorch LSTM
- Part 2: Beam Search Decoder
"""

from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open
import operator
from queue import PriorityQueue

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15


class Vocab:
    """This class handles the mapping between the words and their indicies"""
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, PAD_index: PAD_token}
        self.n_words = 3  # Count SOS, EOS, and PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# Data Loading
######################################################################

def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    """
    logging.info("Reading lines of %s...", input_file)
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """Creates the vocabs for each of the langues based on the training corpus."""
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################
# PART 1: Dataset Class for Batching
######################################################################

class TranslationDataset(Dataset):
    """Dataset class to handle translation pairs for batching"""
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # Convert sentences to index lists
        src_indices = [self.src_vocab.word2index.get(word, PAD_index) 
                       for word in pair[0].split()]
        tgt_indices = [self.tgt_vocab.word2index.get(word, PAD_index) 
                       for word in pair[1].split()]
        
        # Add EOS token
        src_indices.append(EOS_index)
        tgt_indices.append(EOS_index)
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch to the same length.
    Returns padded tensors and their original lengths.
    """
    # Sort batch by source length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    src_seqs, tgt_seqs = zip(*batch)
    
    # Get lengths
    src_lengths = torch.tensor([len(seq) for seq in src_seqs])
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_seqs])
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, 
                                                   padding_value=PAD_index)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, 
                                                   padding_value=PAD_index)
    
    return src_padded, src_lengths, tgt_padded, tgt_lengths


######################################################################
# PART 1: Encoder with PyTorch LSTM
######################################################################

class EncoderRNN(nn.Module):
    """
    Bidirectional LSTM Encoder using PyTorch's built-in LSTM.
    """
    def __init__(self, input_size, hidden_size, embedding_size, num_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=PAD_index)
        
        # PyTorch LSTM: bidirectional=True makes it a BiLSTM
        self.lstm = nn.LSTM(
            embedding_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, input_seqs, input_lengths):
        """
        Args:
            input_seqs: [batch_size, seq_len] or [seq_len] for single sequence
            input_lengths: [batch_size] or scalar
        Returns:
            outputs: [batch_size, seq_len, 2*hidden_size]
            hidden: tuple of (h, c) each [batch_size, hidden_size]
        """
        # Handle single sequence
        if input_seqs.dim() == 1:
            input_seqs = input_seqs.unsqueeze(0)
            input_lengths = torch.tensor([input_lengths], device=device)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = input_seqs.size(0)
        
        # Embed
        embedded = self.embedding(input_seqs)  # [batch_size, seq_len, embedding_size]
        
        # Pack padded sequence for efficiency
        packed = pack_padded_sequence(embedded, input_lengths.cpu(), 
                                     batch_first=True, enforce_sorted=True)
        
        # Run through LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, seq_len, 2*hidden_size]
        
        # hidden and cell: [num_layers*2, batch_size, hidden_size]
        # We use the backward pass from the last layer to initialize decoder
        hidden_backward = hidden[-1]  # [batch_size, hidden_size]
        cell_backward = cell[-1]      # [batch_size, hidden_size]
        
        return outputs, (hidden_backward, cell_backward)


######################################################################
# PART 1: Decoder with PyTorch LSTM
######################################################################

class AttnDecoderRNN(nn.Module):
    """
    Attention-based decoder using PyTorch's LSTM.
    """
    def __init__(self, hidden_size, output_size, num_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # Attention layers (Bahdanau-style)
        self.W_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.U_attn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)

        # PyTorch LSTM (not bidirectional)
        lstm_input_size = self.hidden_size + (self.hidden_size * 2)
        self.lstm = nn.LSTM(lstm_input_size, self.hidden_size, 
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout_p if num_layers > 1 else 0)
        
        # Output layer
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Args:
            input: [batch_size]
            hidden: tuple (h, c) each [num_layers, batch_size, hidden_size]
            encoder_outputs: [batch_size, src_seq_len, 2*hidden_size]
        Returns:
            log_softmax: [batch_size, vocab_size]
            hidden: tuple (h, c) 
            attn_weights: [batch_size, src_seq_len]
        """
        batch_size = input.size(0)
        
        # Embed input
        embedded = self.embedding(input)  # [batch_size, hidden_size]
        embedded = self.dropout(embedded)
        
        # Unpack hidden state
        h_prev, c_prev = hidden
        
        # Calculate attention using the last layer's hidden state
        h_last_layer = h_prev[-1] if h_prev.dim() == 3 else h_prev
        
        w_h = self.W_attn(h_last_layer).unsqueeze(1)  # [batch_size, 1, hidden_size]
        u_e = self.U_attn(encoder_outputs)  # [batch_size, src_seq_len, hidden_size]
        
        attn_scores = self.v_attn(torch.tanh(w_h + u_e)).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, src_seq_len]
        
        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # Concatenate embedding and context
        rnn_input = torch.cat((embedded, context), 1)
        rnn_input = rnn_input.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # Run through LSTM
        output, hidden = self.lstm(rnn_input, hidden)
        output = output.squeeze(1)  # [batch_size, hidden_size]
        
        # Generate output distribution
        output_logits = self.out(output)
        log_softmax = F.log_softmax(output_logits, dim=1)
        
        return log_softmax, hidden, attn_weights


######################################################################
# PART 1: Batched Training Function
######################################################################

def train_batch(src_seqs, src_lengths, tgt_seqs, tgt_lengths, 
                encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    """Train on a batch of sequences."""
    encoder.train()
    decoder.train()
    
    optimizer.zero_grad()
    
    batch_size = src_seqs.size(0)
    max_tgt_len = tgt_seqs.size(1)
    
    # Run encoder
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lengths)
    
    # Initialize decoder hidden state from encoder
    h, c = encoder_hidden
    decoder_hidden = (
        h.unsqueeze(0).repeat(decoder.num_layers, 1, 1),
        c.unsqueeze(0).repeat(decoder.num_layers, 1, 1)
    )
    
    # Start with SOS token
    decoder_input = torch.full((batch_size,), SOS_index, dtype=torch.long, device=device)
    
    loss = 0
    
    # Teacher forcing
    for di in range(max_tgt_len - 1):
        decoder_output, decoder_hidden, _ = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        
        target = tgt_seqs[:, di]
        loss += criterion(decoder_output, target)
        decoder_input = target
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
    
    optimizer.step()
    
    return loss.item() / (max_tgt_len - 1)


######################################################################
# PART 2: Beam Search Implementation
######################################################################

class BeamSearchNode:
    """Node in the beam search tree."""
    def __init__(self, hidden, prev_node, word_id, log_prob, length):
        self.hidden = hidden
        self.prev_node = prev_node
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length
    
    def eval(self, alpha=0.7):
        """Evaluation with length normalization."""
        return self.log_prob / float(self.length + 1e-6) ** alpha
    
    def __lt__(self, other):
        return self.eval() < other.eval()


def beam_search_decode(encoder, decoder, src_tensor, src_length, tgt_vocab, 
                       beam_width=5, max_length=MAX_LENGTH, alpha=0.7):
    """Beam search decoding."""
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode
        encoder_outputs, encoder_hidden = encoder(src_tensor, src_length)
        
        # Initialize decoder
        h, c = encoder_hidden
        decoder_hidden = (
            h.unsqueeze(0).repeat(decoder.num_layers, 1, 1),
            c.unsqueeze(0).repeat(decoder.num_layers, 1, 1)
        )
        
        # Start node
        start_node = BeamSearchNode(
            hidden=decoder_hidden,
            prev_node=None,
            word_id=SOS_index,
            log_prob=0.0,
            length=0
        )
        
        nodes = PriorityQueue()
        nodes.put((-start_node.eval(), start_node))
        end_nodes = []
        
        # Beam search
        for step in range(max_length):
            candidates = []
            
            for _ in range(min(beam_width, nodes.qsize())):
                score, node = nodes.get()
                decoder_input = torch.tensor([node.word_id], device=device)
                
                if node.word_id == EOS_index and node.prev_node is not None:
                    end_nodes.append((score, node))
                    continue
                
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, node.hidden, encoder_outputs
                )
                
                log_probs, indices = decoder_output.topk(beam_width)
                
                for k in range(beam_width):
                    word_id = indices[0][k].item()
                    log_prob = log_probs[0][k].item()
                    
                    new_node = BeamSearchNode(
                        hidden=decoder_hidden,
                        prev_node=node,
                        word_id=word_id,
                        log_prob=node.log_prob + log_prob,
                        length=node.length + 1
                    )
                    
                    candidates.append((-new_node.eval(alpha), new_node))
            
            for item in sorted(candidates, key=operator.itemgetter(0))[:beam_width]:
                nodes.put(item)
            
            if len(end_nodes) >= beam_width:
                break
        
        if len(end_nodes) == 0:
            for _ in range(nodes.qsize()):
                score, node = nodes.get()
                end_nodes.append((score, node))
        
        # Get best hypothesis
        end_nodes = sorted(end_nodes, key=operator.itemgetter(0))
        best_node = end_nodes[0][1]
        
        # Backtrack
        decoded_words = []
        node = best_node
        
        while node.prev_node is not None:
            if node.word_id != EOS_index:
                decoded_words.append(tgt_vocab.index2word[node.word_id])
            node = node.prev_node
        
        decoded_words = decoded_words[::-1]
        
        return decoded_words


######################################################################
# Translation Functions
######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, 
              use_beam_search=False, beam_width=5, max_length=MAX_LENGTH):
    """Translate a sentence."""
    encoder.eval()
    decoder.eval()
    
    # Convert sentence to indices
    src_indices = [src_vocab.word2index.get(word, PAD_index) 
                  for word in sentence.split()]
    src_indices.append(EOS_index)
    
    src_tensor = torch.tensor(src_indices, device=device)
    src_length = len(src_indices)
    
    if use_beam_search:
        decoded_words = beam_search_decode(
            encoder, decoder, src_tensor, src_length, tgt_vocab,
            beam_width=beam_width, max_length=max_length
        )
        return decoded_words, None
    
    # Greedy decoding
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(src_tensor, src_length)
        
        h, c = encoder_hidden
        decoder_hidden = (
            h.unsqueeze(0).repeat(decoder.num_layers, 1, 1),
            c.unsqueeze(0).repeat(decoder.num_layers, 1, 1)
        )
        
        decoder_input = torch.tensor([SOS_index], device=device)
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, len(src_indices))
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            decoder_attentions[di] = decoder_attention.squeeze(0).data
            
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])
            
            decoder_input = topi.squeeze(1).detach()
        
        return decoded_words, decoder_attentions[:di + 1]


def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, 
                       use_beam_search=False, beam_width=5,
                       max_num_sentences=None, max_length=MAX_LENGTH):
    """Translate multiple sentences."""
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab,
                                   use_beam_search=use_beam_search,
                                   beam_width=beam_width,
                                   max_length=max_length)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1,
                              use_beam_search=False, beam_width=5):
    """Translate random sentences."""
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab,
                                   use_beam_search=use_beam_search,
                                   beam_width=beam_width)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        if use_beam_search:
            print(f'  (beam search, width={beam_width})')
        print('')


######################################################################
# Evaluation
######################################################################

def clean(strx):
    """input: string with bpe, EOS; output: cleaned string"""
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


def evaluate_bleu(encoder, decoder, pairs, src_vocab, tgt_vocab,
                 use_beam_search=False, beam_width=5):
    """Evaluate BLEU score."""
    logging.info("Starting translation for BLEU evaluation...")
    translated_sentences = translate_sentences(
        encoder, decoder, pairs, src_vocab, tgt_vocab,
        use_beam_search=use_beam_search, beam_width=beam_width,
        max_num_sentences=len(pairs) # Ensure all pairs are translated
    )
    logging.info("Translation finished. Preparing references and candidates...")

    references = []
    candidates = []

    references = [[clean(pair[1]).split(), ] for pair in pairs[:len(translated_sentences)]]
    candidates = [clean(sent).split() for sent in translated_sentences]
    
    bleu = corpus_bleu(references, candidates)
    
    return bleu


######################################################################
# Visualization
######################################################################

def show_attention(input_sentence, output_words, attentions):
    """Visualize attention mechanism."""
    source_labels = input_sentence.split(' ')
    target_labels = output_words
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.matshow(attentions.cpu().numpy(), cmap='gray')
    
    ax.set_xticks(np.arange(len(source_labels)))
    ax.set_yticks(np.arange(len(target_labels)))
    ax.set_xticklabels(source_labels, rotation=90)
    ax.set_yticklabels(target_labels)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(source_labels))))
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(target_labels))))
    
    output_filename = 'attention_visualization.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Attention plot saved to {output_filename}")
    plt.close(fig)


def translate_and_show_attention(input_sentence, encoder, decoder, src_vocab, tgt_vocab):
    """Translate and visualize attention."""
    output_words, attentions = translate(
        encoder, decoder, input_sentence, src_vocab, tgt_vocab,
        use_beam_search=False
    )
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    if attentions is not None:
        show_attention(input_sentence, output_words, attentions)


######################################################################
# Benchmarking
######################################################################

def benchmark_training_speed(encoder, decoder, train_pairs, src_vocab, tgt_vocab,
                             batch_size, n_sentences_to_process=1024):
    """Benchmark training speed for the batched model."""
    import math # Import math for ceiling division
    
    # Calculate the minimum number of batches to process
    n_batches = math.ceil(n_sentences_to_process / batch_size)
    
    logging.info(f"Running benchmark for {n_batches} batches to process at least {n_sentences_to_process} sentences...")

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    criterion = nn.NLLLoss(ignore_index=PAD_index)
    
    encoder.train()
    decoder.train()
    
    start_time = time.time()
    total_examples = 0
    
    # Loop for just the required number of batches
    for i, (src_seqs, src_lengths, tgt_seqs, tgt_lengths) in enumerate(train_loader):
        if i >= n_batches:
            break
            
        src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
        
        train_batch(src_seqs, src_lengths, tgt_seqs, tgt_lengths,
                   encoder, decoder, optimizer, criterion)
        
        total_examples += src_seqs.size(0)

    elapsed_time = time.time() - start_time
    sentences_per_sec = total_examples / elapsed_time
    
    logging.info(f"Benchmark processed {total_examples} sentences in {elapsed_time:.2f} seconds.")
    return sentences_per_sec


######################################################################
# Training Loop
######################################################################

def train_iters(encoder, decoder, train_pairs, src_vocab, tgt_vocab, args):
    """Training loop with batching."""
    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_index)
    
    # Load dev set for evaluation
    dev_pairs = split_lines(args.dev_file)
    
    start = time.time()
    print_loss_total = 0
    iter_num = 0
    
    n_epochs = args.n_iters // len(train_loader) + 1
    
    for epoch in range(n_epochs):
        for src_seqs, src_lengths, tgt_seqs, tgt_lengths in train_loader:
            iter_num += 1
            if iter_num > args.n_iters:
                break
            
            src_seqs = src_seqs.to(device)
            src_lengths = src_lengths.to(device)
            tgt_seqs = tgt_seqs.to(device)
            tgt_lengths = tgt_lengths.to(device)
            
            loss = train_batch(src_seqs, src_lengths, tgt_seqs, tgt_lengths,
                             encoder, decoder, optimizer, criterion)
            
            print_loss_total += loss
            
            if iter_num % args.checkpoint_every == 0:
                state = {'iter_num': iter_num,
                        'enc_state': encoder.state_dict(),
                        'dec_state': decoder.state_dict(),
                        'opt_state': optimizer.state_dict(),
                        'src_vocab': src_vocab,
                        'tgt_vocab': tgt_vocab,
                        }
                filename = 'state_%010d.pt' % iter_num
                torch.save(state, filename)
                logging.debug('wrote checkpoint to %s', filename)
            
            if iter_num % args.print_every == 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                           time.time() - start,
                           iter_num,
                           iter_num / args.n_iters * 100,
                           print_loss_avg)
                
                # Translate from dev set
                translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
                
                # Evaluate BLEU
                dev_bleu = evaluate_bleu(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)
                logging.info('Dev BLEU score: %.2f', dev_bleu)
        
        if iter_num > args.n_iters:
            break


######################################################################
# Main Function
######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder')
    ap.add_argument('--embedding_size', default=256, type=int,
                    help='embedding size for encoder')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--batch_size', default=32, type=int,
                    help='batch size for training')
    ap.add_argument('--print_every', default=1000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code')
    ap.add_argument('--tgt_lang', default='en',
                    help='Target (output) language code')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    
    # Part 1 options
    ap.add_argument('--benchmark_only', action='store_true',
                    help='only run the benchmark and exit')
    ap.add_argument('--translate_only', action='store_true',
                    help='run translation on the test set and exit')
    ap.add_argument('--n_layers', default=1, type=int,
                    help='number of layers in the encoder and decoder LSTMs')
    ap.add_argument('--dropout', default=0.1, type=float,
                    help='dropout probability for LSTM layers')
    ap.add_argument('--use_beam_search', action='store_true',
                    help='use beam search for decoding during translation')
    ap.add_argument('--beam_width', default=5, type=int,
                    help='beam width for beam search')

    args = ap.parse_args()
    logging.info(args)

    # Load vocabs and/or checkpoint
    if args.load_checkpoint:
        logging.info('loading checkpoint from %s', args.load_checkpoint[0])
        checkpoint = torch.load(args.load_checkpoint[0],
                                map_location=device,
                                weights_only=False)
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
    else:
        # If not loading from checkpoint, create vocabs from training data
        src_vocab, tgt_vocab = make_vocabs(args.src_lang, args.tgt_lang,
                                           args.train_file)

    # Initialize models
    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size, args.embedding_size,
                         num_layers=args.n_layers, dropout=args.dropout).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words,
                             num_layers=args.n_layers, dropout_p=args.dropout).to(device)

    # Load model state from checkpoint
    if args.load_checkpoint:
        encoder.load_state_dict(checkpoint['enc_state'])
        decoder.load_state_dict(checkpoint['dec_state'])
        # Note: optimizer state is saved but train_iters re-initializes it.
        # For a more robust resume, the optimizer should also be loaded.

    logging.info('Encoder: %s', encoder)
    logging.info('Decoder: %s', decoder)

    # Handle different execution modes
    if args.benchmark_only:
        train_pairs = split_lines(args.train_file)
        SENTENCES_TO_BENCHMARK = 1024
        logging.info("Benchmarking training speed...")
        sentences_per_sec = benchmark_training_speed(
            encoder, decoder, train_pairs, src_vocab, tgt_vocab,
            args.batch_size, n_sentences_to_process=SENTENCES_TO_BENCHMARK
        )
        print("Benchmark complete.\n")
        print(f"Training Speed (batch_size={args.batch_size}): {sentences_per_sec:.2f} sentences/sec")
        return

    if args.translate_only:
        if not args.load_checkpoint:
            logging.error("A checkpoint must be provided for translation. Use --load_checkpoint.")
            return

        test_pairs = split_lines(args.test_file)
        logging.info(f"Translating {len(test_pairs)} sentences from {args.test_file}...")

        # Translate all sentences in the test file
        translated_sentences = translate_sentences(
            encoder, decoder, test_pairs, src_vocab, tgt_vocab,
            use_beam_search=args.use_beam_search, beam_width=args.beam_width
        )

        # Write translations to output file
        with open(args.out_file, 'w', encoding='utf-8') as f:
            for sent in translated_sentences:
                f.write(clean(sent) + '\n')
        logging.info(f"Translations written to {args.out_file}")

        # Evaluate BLEU score on the test set
        test_bleu = evaluate_bleu(
            encoder, decoder, test_pairs, src_vocab, tgt_vocab,
            use_beam_search=args.use_beam_search, beam_width=args.beam_width
        )
        logging.info(f"Test BLEU score: {test_bleu * 100:.2f}")

        # Visualize attention for one sentence if not using beam search
        if not args.use_beam_search:
            random_pair = random.choice(test_pairs)
            logging.info("Visualizing attention for a random test sentence...")
            translate_and_show_attention(random_pair[0], encoder, decoder, src_vocab, tgt_vocab)

        return

    # Default to training
    train_pairs = split_lines(args.train_file)
    logging.info("Starting or resuming training...")
    train_iters(encoder, decoder, train_pairs, src_vocab, tgt_vocab, args)


if __name__ == '__main__':
    main()