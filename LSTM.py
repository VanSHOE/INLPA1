import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import nltk  # for tokenizing sentences
from smoothing import get_token_list, sentence_tokenizer
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import numpy as np

# nltk.download('punkt')

sentenceLens = {}

device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"


class Data(torch.utils.data.Dataset):
    def __init__(self, entireText: str):
        self.sentences = sentence_tokenizer(entireText)
        self.device = device
        self.Ssentences = [sentence for sentence in self.sentences if 40 > len(sentence) > 0]
        self.Lsentences = [sentence for sentence in self.sentences if len(sentence) >= 40]
        # split long sentences into 2
        self.sentences = []
        for sentence in self.Lsentences:
            half = len(sentence) // 2
            self.sentences.append(sentence[:half])
            self.sentences.append(sentence[half:])
        self.sentences += self.Ssentences

        self.sentences = [sentence for sentence in self.sentences if len(sentence) <= 40]

        print(self.sentences)
        print(len(self.sentences))
        self.vocab = set()
        self.mxSentSize = 0

        for sentence in self.sentences:
            if len(sentence) not in sentenceLens:
                sentenceLens[len(sentence)] = 1
            else:
                sentenceLens[len(sentence)] += 1
            for token in sentence:
                self.vocab.add(token)

            if len(sentence) > self.mxSentSize:
                self.mxSentSize = len(sentence)

        self.vocab = list(self.vocab)

        # add padding token
        self.vocab.append("<pad>")
        # add Unknown
        self.vocab.append("<unk>")

        self.vocabSet = set(self.vocab)
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        # pad each sentence to 40
        for i in range(len(self.sentences)):
            self.sentences[i] = ["<pad>"] * (self.mxSentSize - len(self.sentences[i])) + self.sentences[i]

        self.sentencesIdx = torch.tensor([[self.w2idx[token] for token in sentence] for sentence in self.sentences],
                                         device=self.device)

    def __len__(self):
        return len(self.sentencesIdx)

    def __getitem__(self, idx):
        # sentence, last word
        return self.sentencesIdx[idx][:-1], self.sentencesIdx[idx][1:]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, vocab_size):
        super(LSTM, self).__init__()  # call the init function of the parent class
        self.device = device
        self.num_layers = num_layers  # number of LSTM layers
        self.hidden_size = hidden_size  # size of LSTM hidden state
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.decoder = nn.Linear(hidden_size, vocab_size)  # linear layer to map the hidden state to output classes

        self.elayer = nn.Embedding(vocab_size, input_size)

        self.to(self.device)

    def forward(self, x, state=None):
        # Set initial states for the LSTM layer or use the states passed from the previous time step
        embeddings = self.elayer(x)

        # Forward propagate through the LSTM layer
        out, _ = self.lstm(embeddings)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Pass the extracted output through the linear layer to map it to output classes
        # decode for all time steps
        # for i in range(out.size(1)):
        #     out[:, i, :] = self.decoder(out[:, i, :])

        return self.decoder(out)


def train(model, data, optimizer, criterion, valDat):
    epoch_loss = 0
    model.train()

    dataL = DataLoader(data, batch_size=32, shuffle=True)
    lossDec = True
    prevLoss = 10000000
    prevValLoss = 10000000
    epoch = 0
    es_patience = 2
    while lossDec:
        epoch_loss = 0
        for i, (x, y) in enumerate(dataL):

            optimizer.zero_grad()
            x = x.to(model.device)

            y = y.to(model.device)

            output = model(x)

            y = y.view(-1)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # print loss every 100 batches
            if i % 100 == 0:
                print(f"Epoch {epoch + 1} Batch {i} loss: {loss.item()}")

        validationLoss = getLossDataset(valDat, model)
        print(f"Validation loss: {validationLoss}")
        if validationLoss > prevValLoss:
            print("Validation loss increased")
            if es_patience > 0:
                es_patience -= 1

            else:  # early stopping
                print("Early stopping")
                lossDec = False
        prevValLoss = validationLoss
        model.train()
        if epoch_loss / len(dataL) > prevLoss:
            lossDec = False
        prevLoss = epoch_loss / len(dataL)

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")
        epoch += 1


def perplexity(data, model, sentence):
    sentence = get_token_list(sentence)
    for tokenIdx in range(len(sentence)):
        if sentence[tokenIdx] not in data.vocabSet:
            sentence[tokenIdx] = "<unk>"

    sentence = torch.tensor([data.w2idx[token] for token in sentence], device=model.device)
    y = model(sentence[:-1])
    # print(y.shape)
    probs = torch.nn.functional.softmax(y, dim=-1).cpu().detach().numpy()
    target = sentence[1:]
    perp = 0
    for i in range(len(target)):
        perp += -np.log(probs[i][target[i]])
    return np.exp(perp / len(target.cpu().numpy()))


def getPerpDataset(data: Data):
    model.eval()

    # check perplexity for each sentence in data
    perp = 0
    perps = {}
    with alive_bar(len(data.sentences)) as bar:
        for sentence in data.sentences:
            newPerp = perplexity(data, model, ' '.join(sentence))
            perp += newPerp
            if newPerp in perps:
                perps[newPerp] += 1
            else:
                perps[newPerp] = 1
            bar()

    # print(perps)
    # histogram binned
    plt.hist(perps.keys(), bins=100)
    plt.show()
    print(f"Mean: {perp / len(data.sentences)}")
    # print median taking into account the freq
    perpList = []
    for perp in perps:
        perpList += [perp] * perps[perp]
    print(f"Median: {np.median(perpList)}")


def getLossDataset(data: Data, model):
    model.eval()

    dataL = DataLoader(data, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    loss = 0

    for i, (x, y) in enumerate(dataL):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])

        loss += criterion(output, y).item()

    return loss / len(dataL)


if __name__ == '__main__':
    fullText = open("./corpus/Pride and Prejudice - Jane Austen.txt", "r", encoding='utf-8').read()
    # split train test validation
    trainText = fullText[:int(len(fullText) * 0.8)]
    testText = fullText[int(len(fullText) * 0.8):int(len(fullText) * 0.9)]
    valText = fullText[int(len(fullText) * 0.9):]

    train_data = Data(trainText)
    test_data = Data(testText)
    val_data = Data(valText)
    # split data

    model = LSTM(500, 500, 1, len(train_data.vocab), len(train_data.vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, train_data, optimizer, criterion, val_data)

    getPerpDataset(train_data)
    getPerpDataset(val_data)
    getPerpDataset(test_data)
