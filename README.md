# RNN and LSTM Reinvention

This notebook is a small first-principles project for understanding why RNNs and LSTMs were invented.

The goal is not to build a huge model. The goal is to clearly see the difference between:

- a model with no real sequence memory
- a vanilla RNN with hidden-state memory
- an LSTM with gated long-term memory

## Project Idea

The notebook uses a small synthetic memory dataset.

Each sequence looks something like this:

```text
START RED dog tree cloud BLUE music chair QUERY
```

The rule is:

```text
The first RED/BLUE token after START is the correct answer.
Ignore later RED/BLUE tokens because they are distractors.
```

So in the example above, the label is:

```text
RED
```

This creates a simple memory test: can the model remember important information from the beginning of the sequence until the end?

## Dataset

The dataset is stored in a CSV file:

```text
synthetic_memory_500_very_long.csv
```

It contains:

- 500 total examples
- 400 training examples
- 100 validation examples
- 30 tokens per sequence
- labels: RED or BLUE

Important columns include:

- `sequence`: the token sequence
- `label`: numeric label
- `label_name`: RED or BLUE
- `split`: train or val
- `distractor_count`: number of distracting later color tokens

## Models

The notebook compares three models.

### 1. FixedWindowMLP

This model only looks at the last few tokens before `QUERY`.

It does not have real sequence memory, so it struggles when the answer is near the beginning of the sequence.

### 2. VanillaRNNFromScratch

This model manually implements a vanilla RNN without using `nn.RNN`.

The main idea is:

```text
new hidden state = current token information + previous hidden state
```

The final hidden state is used to predict the label.

### 3. LSTMFromScratch

This model manually implements an LSTM without using `nn.LSTM`.

The LSTM uses:

- hidden state
- cell state
- forget gate
- input gate
- candidate cell
- output gate

The main idea is that the LSTM has a controlled memory system, which helps it preserve important information over longer sequences.

## Why This Matters

A vanilla RNN has memory, but that memory can get overwritten as more tokens appear.

An LSTM improves this by adding gates that control:

- what old memory to forget
- what new memory to write
- what memory to expose for prediction

This is why LSTMs were important before Transformers became dominant.

## What I Learned

Through this notebook, I practiced:

- loading a CSV dataset in Kaggle
- tokenizing sequence data
- building a vocabulary
- converting tokens to tensors
- using `TensorDataset` and `DataLoader`
- implementing a vanilla RNN from scratch
- implementing an LSTM from scratch
- training sequence models with PyTorch
- comparing validation accuracy
- inspecting model predictions and LSTM gate values

## Tools Used

- Python
- PyTorch
- Pandas
- Matplotlib
- Kaggle Notebooks

## Main Takeaway

```text
FixedWindowMLP = no real sequence memory
Vanilla RNN = hidden-state memory
LSTM = gated long-term memory
Transformer = later replaces step-by-step memory with attention
```

**Unfortunately RNN and LSTM actually showed very similar results
