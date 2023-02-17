import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import nltk  # for tokenizing sentences
from smoothing import get_token_list, sentence_tokenizer
import matplotlib.pyplot as plt

# nltk.download('punkt')

sentenceLens = {}

device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"


class Data(torch.utils.data.Dataset):
    def __init__(self, entireText: str):
        # self.sentences = nltk.sent_tokenize(entireText)
        self.sentences = sentence_tokenizer(entireText)
        # print(self.sentences)
        # plot bar graph of lengths of sentences
        # plt.hist([len(sentence) for sentence in self.sentences], bins=50)
        # plt.show()
        #
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
        # print total sentences and those with len > 40
        # print("Total sentences:", len(self.sentences))
        # print("Sentences with len > 40:", len(self.Lsentences))
        # print("Sentences with len < 40:", len(self.Ssentences))

        self.sentences = [sentence for sentence in self.sentences if len(sentence) <= 40]

        # exit(33)
        # plt.hist([len(sentence) for sentence in self.sentences], bins=50)
        # plt.show()
        # exit(33)
        # sort by length
        # self.sentences.sort(key=lambda x: len(x))
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
        self.vocabSet = set(self.vocab)
        # add padding token
        self.vocab.append("<PAD>")
        # add Unknown
        self.vocab.append("<UNK>")
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        # pad each sentence to 40
        for i in range(len(self.sentences)):
            self.sentences[i] = ["<PAD>"] * (self.mxSentSize - len(self.sentences[i])) + self.sentences[i]

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


def train(model, data, optimizer, criterion):
    epoch_loss = 0
    model.train()

    dataL = DataLoader(data, batch_size=32, shuffle=True)

    for epoch in range(1):
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

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")


def perplexity(data, model, sentence):
    sentence = get_token_list(sentence)
    sentence = torch.tensor([data.w2idx[token] for token in sentence], device=model.device)
    y = model(sentence[:-1])
    probs = torch.nn.functional.softmax(y, dim=-1)
    target = sentence[1:]
    word_probs = torch.gather(probs, 1, target.view(-1, 1)).squeeze()
    perplexity = torch.exp(-torch.log(word_probs).sum() / len(sentence))
    return perplexity.item()


if __name__ == '__main__':
    data = Data(open("./corpus/Ulysses - James Joyce.txt", "r", encoding='utf-8').read())
    model = LSTM(500, 500, 1, len(data.vocab), len(data.vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, data, optimizer, criterion)

    model.eval()
    context = get_token_list("Mr Dedalus said")
    textCopy = context.copy()
    context = torch.tensor([data.w2idx[token] for token in context], device=model.device)
    # perplexity(data, model, context)
    # exit(55)
    y = model(context)
    # print word with max prob
    print(y.shape)
    print(y[-1].shape)

    print(perplexity(data, model, "Mr Dedalus said"))
