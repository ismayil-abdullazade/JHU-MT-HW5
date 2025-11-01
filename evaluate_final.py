#!/usr/bin/env python3

import torch
from seq2seq import *

def main():
    print("Loading best model...")
    
    # Load the best model
    state = torch.load('state_0000060017.pt', weights_only=False)
    src_vocab, tgt_vocab = state['src_vocab'], state['tgt_vocab']
    
    # Initialize models
    encoder = EncoderRNN(src_vocab.n_words, 512, num_layers=2, dropout=0.3).to(device)
    decoder = AttnDecoderRNN(512, tgt_vocab.n_words, num_layers=2, dropout_p=0.3).to(device)
    
    encoder.load_state_dict(state['enc_state'])
    decoder.load_state_dict(state['dec_state'])
    
    # Load dev data
    dev_pairs = split_lines('data/fren.dev.bpe')
    
    print("Evaluating on dev set with beam size 10...")
    translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, max_num_sentences=100, beam_size=10)
    references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
    candidates = [clean(sent).split() for sent in translated_sentences]
    dev_bleu = corpus_bleu(references, candidates) * 100
    print(f'Final Dev BLEU score (beam=10): {dev_bleu:.2f}')
    
    print("\nSample translations:")
    for i in range(5):
        pair = dev_pairs[i]
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, beam_size=10)
        output_sentence = ' '.join(output_words)
        print(f'Source: {pair[0]}')
        print(f'Target: {pair[1]}')
        print(f'Output: {clean(output_sentence)}')
        print()

if __name__ == '__main__':
    main()