import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import nltk  # for tokenizing sentences
from smoothing import get_token_list, sentence_tokenizer, rem_low_freq
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import numpy as np
import random
import time

MODEL = "LM6"

# nltk.download('punkt')

sentenceLens = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32


# device = "cpu"


class Data(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.device = device

        self.cutoff = 20

        self.Ssentences = [sentence for sentence in self.sentences if self.cutoff > len(sentence) > 0]
        self.Lsentences = [sentence for sentence in self.sentences if len(sentence) >= self.cutoff]
        # split long sentences into 2
        self.sentences = []
        for sentence in self.Lsentences:
            half = len(sentence) // 2
            self.sentences.append(sentence[:half])
            self.sentences.append(sentence[half:])
        self.sentences += self.Ssentences

        self.sentences = [sentence for sentence in self.sentences if len(sentence) <= self.cutoff]

        # print(self.sentences)
        print(len(self.sentences))
        self.vocab = set()
        self.mxSentSize = 0
        # self.mxSentSize = 20
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
        if "<unk>" not in self.vocab:
            self.vocab.append("<unk>")

        self.vocabSet = set(self.vocab)
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        # pad each sentence to 40
        for i in range(len(self.sentences)):
            # print(type(self.sentences[i]))
            self.sentences[i] = ["<pad>"] * (self.mxSentSize - len(self.sentences[i])) + self.sentences[i]

        self.sentencesIdx = torch.tensor([[self.w2idx[token] for token in sentence] for sentence in self.sentences],
                                         device=self.device)

    def handle_unknowns(self, vocab_set, vocab):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j] not in vocab_set:
                    # remove from vocab and vocab set
                    if self.sentences[i][j] in self.vocab:
                        self.vocab.remove(self.sentences[i][j])
                    if self.sentences[i][j] in self.vocabSet:
                        self.vocabSet.remove(self.sentences[i][j])
                    self.sentences[i][j] = "<unk>"
        self.w2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2w = {i: w for i, w in enumerate(vocab)}
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
        self.train_data = None

        self.elayer = nn.Embedding(vocab_size, input_size)

        self.to(self.device)

    def forward(self, x, state=None):
        # Set initial states for the LSTM layer or use the states passed from the previous time step
        embeddings = self.elayer(x)

        # Forward propagate through the LSTM layer
        out, _ = self.lstm(embeddings)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return self.decoder(out)


def train(model, data, optimizer, criterion, valDat, maxPat=5):
    epoch_loss = 0
    model.train()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    lossDec = True
    prevLoss = 10000000
    prevValLoss = 10000000
    epoch = 0
    es_patience = maxPat
    model.train_data = data
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
            # if i % 100 == 0:
            #     print(f"Epoch {epoch + 1} Batch {i} loss: {loss.item()}")

        validationLoss = getLossDataset(valDat, model)
        print(f"Validation loss: {validationLoss}")
        if validationLoss - epoch_loss / len(dataL) > 2:
            print("Validation loss increased")
            if es_patience > 0:
                es_patience -= 1

            else:  # early stopping
                print("Early stopping")
                model.load_state_dict(torch.load(f"{MODEL}.pt"))
                lossDec = False
        else:
            torch.save(model.state_dict(), f"{MODEL}.pt")
            es_patience = maxPat
        prevValLoss = validationLoss
        model.train()
        if epoch_loss / len(dataL) > prevLoss:
            lossDec = False
        prevLoss = epoch_loss / len(dataL)

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")
        epoch += 1


def perplexity(data, model, sentence):
    sentence = get_token_list(sentence)
    if model.train_data is None:
        print("No training data")
        return
    for tokenIdx in range(len(sentence)):
        if sentence[tokenIdx] not in model.train_data.vocabSet:
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


def getPerpDataset(data: Data, filename: str):
    model.eval()

    # check perplexity for each sentence in data
    perp = 0
    perps = {}

    toWrite = []
    with alive_bar(len(data.sentences)) as bar:
        for sentence in data.sentences:
            newPerp = perplexity(data, model, ' '.join(sentence))
            # sentence<tab>perp
            toWrite.append(f"{' '.join(sentence)}\t{newPerp}")
            perp += newPerp
            if newPerp in perps:
                perps[newPerp] += 1
            else:
                perps[newPerp] = 1
            bar()

    output = open(filename, "w", encoding="utf-8")
    output.write(f"{perp / len(data.sentences)}\n")
    # write sentences
    output.write('\n'.join(toWrite))
    output.close()
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


def rem_low_freq_sentences(sentences, freq):
    dist = {}
    for sentence in sentences:
        for token in sentence:
            if token in dist:
                dist[token] += 1
            else:
                dist[token] = 1

    # replace with unk
    for sentence in sentences:
        for tokenIdx in range(len(sentence)):
            if dist[sentence[tokenIdx]] < freq:
                sentence[tokenIdx] = "<unk>"

    return sentences


def getLossDataset(data: Data, model):
    model.eval()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

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
    # seed random with time
    random.seed(time.time())

    fullText = open("./corpus/Ulysses - James Joyce.txt", "r", encoding='utf-8').read().lower()
    sentences = sentence_tokenizer(fullText, -1)

    sentences = [sentence for sentence in sentences if len(sentence) > 0]
    # split train test validation using random
    trainText = []
    testText = []
    valText = []
    for sentence in sentences:
        rand = random.random()
        if rand < 0.6:
            trainText.append(sentence)
        elif rand < 0.8:
            testText.append(sentence)
        else:
            valText.append(sentence)

    trainText = rem_low_freq_sentences(trainText, 3)

    train_data = Data(trainText)
    test_data = Data(testText)
    val_data = Data(valText)
    # print sizes of all vocab
    # exit(333)
    # split data

    model = LSTM(300, 300, 1, len(train_data.vocab), len(train_data.vocab))
    test_data.handle_unknowns(train_data.vocabSet, train_data.vocab)
    val_data.handle_unknowns(train_data.vocabSet, train_data.vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, train_data, optimizer, criterion, val_data, 4)

    getPerpDataset(val_data, "val.log")
    getPerpDataset(test_data, f"2020115006_{MODEL}_test-perplexity.txt")

    getPerpDataset(train_data, f"2020115006_{MODEL}_train-perplexity.txt")
