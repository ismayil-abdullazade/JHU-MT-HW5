# Neural Machine Translation with Beam Search

This directory contains the code for an advanced sequence-to-sequence (Seq2Seq) model for machine translation, implemented in PyTorch. It is designed to translate text from French to English.

The implementation includes modern NMT features such as batched training for efficiency and a beam search decoder for improved translation quality.

## Code and Files

-   `seq2seq.py`: The main executable Python script containing the entire model architecture, data handling, and training/inference logic.
-   `data/`: Directory that must contain the BPE-tokenized datasets (`fren.train.bpe`, `fren.dev.bpe`, `fren.test.bpe`).

## Core Features Implemented

-   **Model Architecture:** A bidirectional LSTM encoder with a Bahdanau attention-based LSTM decoder.
-   **Data Processing:** Custom `Dataset` and `collate_fn` for efficient batching, padding, and packing of sequences.
-   **Training:** Batched training loop with teacher forcing, gradient clipping, and periodic checkpointing.
-   **Decoding Algorithms:**
    1.  **Greedy Search:** Baseline decoder that selects the most probable word at each step.
    2.  **Beam Search:** Advanced decoder that maintains multiple translation hypotheses to find a more globally optimal sequence.

## Setup Instructions

1.  **Dependencies:** This script requires Python 3.8+ and several packages. Install them using pip:
    ```bash
    pip install torch nltk matplotlib numpy
    ```

2.  **Data Files:** Place your training, development, and test data files inside a subdirectory named `data/`. The script expects the following files by default:
    -   `data/fren.train.bpe`
    -   `data/fren.dev.bpe`
    -   `data/fren.test.bpe`

## Script Usage and Command-Line Arguments

The `seq2seq.py` script is controlled via command-line arguments. The primary modes of operation are training, translation, and benchmarking.

### **1. Training a Model**

To train a new model from scratch, run the script without `--translate_only` or `--benchmark_only`.

**Basic Training Command:**
```bash
python3 seq2seq.py --n_iters 100000 --batch_size 32
```

**Common Training Arguments:**

-   `--n_iters <int>`: Total number of training iterations (batches) to run. Default: `100000`.
-   `--batch_size <int>`: Number of sentences in each training batch. Default: `32`.
-   `--hidden_size <int>`: The size of the LSTM hidden states. Default: `256`.
-   `--embedding_size <int>`: The size of the word embeddings. Default: `256`.
-   `--initial_learning_rate <float>`: The learning rate for the Adam optimizer. Default: `0.001`.
-   `--print_every <int>`: Log the training loss every N iterations. Default: `1000`.
-   `--checkpoint_every <int>`: Save a model checkpoint (`state_*.pt`) every N iterations. Default: `10000`.
-   `--load_checkpoint <file>`: Resume training from a previously saved checkpoint file.

### **2. Generating Translations**

Use the `--translate_only` flag to run inference with a trained model. This requires providing a checkpoint file.

**Greedy Search Translation (Default):**
```bash
python3 seq2seq.py --translate_only \
                      --load_checkpoint state_000100000.pt \
                      --test_file data/fren.dev.bpe \
                      --out_file greedy_translations.txt
```

**Beam Search Translation:**
```bash
python3 seq2seq.py --translate_only \
                      --load_checkpoint state_000100000.pt \
                      --test_file data/fren.dev.bpe \
                      --out_file beam_translations.txt \
                      --use_beam_search \
                      --beam_width 5
```

**Translation Arguments:**
-   `--translate_only`: Activates translation mode.
-   `--load_checkpoint <file>`: (Required) Path to the trained model checkpoint (`.pt` file).
-   `--test_file <file>`: The source file to translate. Default: `data/fren.test.bpe`.
-   `--out_file <file>`: The file to write the generated translations to. Default: `out.txt`.
-   `--use_beam_search`: Flag to enable the beam search decoder.
-   `--beam_width <int>`: Sets the beam width for beam search. Default: `5`.

### **3. Benchmarking Performance**

To quickly measure the training speed of the model in sentences per second, use the `--benchmark_only` flag. This will run a short training simulation and exit.

**Basic Benchmark Command:**
```bash
python3 seq2seq.py --benchmark_only --batch_size 32
```
This is useful for quickly assessing the performance impact of different batch sizes or hardware environments.
