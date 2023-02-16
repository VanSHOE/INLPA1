import torch
import torch.nn as nn
import nltk  # for tokenizing sentences
from smoothing import get_token_list
import matplotlib.pyplot as plt

nltk.download('punkt')

sentenceLens = {}


class Data(torch.utils.data.Dataset):
    def __init__(self, entireText: str):
        self.sentences = nltk.sent_tokenize(entireText)
        self.sentences = [sentence.lower() for sentence in self.sentences if 100 > len(sentence) > 0]
        self.token_list = [get_token_list(sentence) for sentence in self.sentences]
        print(len(self.sentences))
        self.vocab = set()
        self.mxSentSize = 0

        for sentence in self.token_list:
            if len(sentence) not in sentenceLens:
                sentenceLens[len(sentence)] = 1
            else:
                sentenceLens[len(sentence)] += 1
            for token in sentence:
                self.vocab.add(token)

            if len(sentence) > self.mxSentSize:
                self.mxSentSize = len(sentence)

        self.vocab = list(self.vocab)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()  # call the init function of the parent class
        self.num_layers = num_layers  # number of LSTM layers
        self.hidden_size = hidden_size  # size of LSTM hidden state
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.decoder = nn.Linear(hidden_size, num_classes)  # linear layer to map hidden state to output classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # check if GPU is available

        self.ebSize = 300
        self.elayer = nn.Embedding(10000, self.ebSize)

    def forward(self, x, state=None):
        # Set initial states for the LSTM layer or use the states passed from the previous time step
        if state is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # initial hidden state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # initial cell state
        else:
            h0, c0 = state

        # Forward propagate through the LSTM layer
        out, _ = self.lstm(x, (
            h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size) and we take the last sequence step

        # Extract the output from the last time step for each input sequence in the batch
        out = out[:, -1, :]  # out: tensor of shape (batch_size, hidden_size)

        # Pass the extracted output through the linear layer to map it to output classes
        out = self.decoder(out)  # out: tensor of shape (batch_size, num_classes)

        return out


data = Data(open("./corpus/Pride and Prejudice - Jane Austen.txt", "r").read())
print(data.mxSentSize)

plt.bar(sentenceLens.keys(), sentenceLens.values())
plt.show()
