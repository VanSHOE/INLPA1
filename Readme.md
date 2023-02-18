# README

This repository contains two files, `LSTM.py` and `smoothing.py`, which are Python scripts for calculating perplexities
of language models.

## LSTM

`LSTM.py` is a script for calculating the perplexity of a language model implemented using Long Short-Term Memory (LSTM)
neural networks. The script takes a path to a `.pth` file containing the trained model as an argument. When called with
an argument, it prompts the user to enter a sentence, and then outputs the perplexity of the sentence according to the
specified model.

### Usage

To run `LSTM.py` with an argument:

`python LSTM.py /path/to/model.pth`

It will then ask you to enter a sentence, and then output the perplexity of the sentence according to the specified
smoothing method.
To generate perplexity files for both the training and testing sets without an argument:

`python LSTM.py`

## Smoothing

`smoothing.py` is a script for calculating the perplexity of a language model with smoothing applied. The script takes
two arguments: the type of smoothing to apply (either "kneser-ney" or "written-bell"), and the name of the corpus file
to use.

### Usage

To run `smoothing.py` with arguments:

`python smoothing.py k /path/to/corpus.txt` or `python smoothing.py w /path/to/corpus.txt`
It will then ask you to enter a sentence, and then output the perplexity of the sentence according to the specified
model.
To generate perplexity files for both the training and testing sets without arguments:

`python smoothing.py`

Note that the corpus files used in `smoothing.py` must be in plain text format and the program will tokenize it into
sentences.


