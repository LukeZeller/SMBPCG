import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import numpy as np
import os

random.seed(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('--niter', type=int, default=300,
                    help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--edim', type=int, default=256,
                    help='number of embedding dimensions')
parser.add_argument('--hdim', type=int, default=256,
                    help='number of hidden dimensions')
parser.add_argument('--tsize', type=int, default=100,
                    help='number of training sets create per level')
parser.add_argument('--probp', type=int, default=30,
                    help='prob a pipe tile is changed')
parser.add_argument('--probs', type=int, default=30,
                    help='prob a stair tile is changed')
opt = parser.parse_args()

assert torch.cuda.is_available()
torch.cuda.set_device(0)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    res = torch.tensor(idxs, dtype=torch.long).cuda()
    return res


pipe = ["<", ">", "[", "]"]
pieces = ["X", "-", "<", ">", "[", "]"]
piecesFull = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]"]
tileMapping = {"X": 0, "S": 1, "-": 2, "?": 3, "Q": 4,
               "E": 5, "<": 6, ">": 7, "[": 8, "]": 9}


def prepareData():
    training_data = []
    testing_data = []
    for filename in os.listdir(opt.directory):
        levelByColumn = []
        with open(opt.directory + "/" + filename) as textFile:
            levelByRow = np.array([list(line) for line in textFile])
        levelByRow = levelByRow[:, :-1]
        # We read in a text file version the level into a 2-D array
        # However, the LSTM will read the level column by column
        # So, it is necessary to swap rows and columns and then flatten the level
        for j in range(len(levelByRow[0])):
            for i in range(len(levelByRow)):
                levelByColumn.append(levelByRow[i][j])
        # Create Training Data
        for k in range(opt.tsize):
            perturbedLevel = []
            for i in range(len(levelByColumn)):
                if levelByColumn[i] in pipe and random.randint(0, 99) < opt.probp:
                    perturbedLevel.append(random.choice(pieces))
                elif levelByColumn[i] == "X" and i % 14 != 13 and random.randint(0, 99) < opt.probs:
                    perturbedLevel.append(random.choice(pieces))
                else:
                    perturbedLevel.append(levelByColumn[i])
            training_data.append((perturbedLevel, levelByColumn))
        # Create Testing Data
        for k in range(int(opt.tsize / 10)):
            perturbedLevel = []
            for i in range(len(levelByColumn)):
                if levelByColumn[i] in pipe and random.randint(0, 99) < opt.probp:
                    perturbedLevel.append(random.choice(pieces))
                elif levelByColumn[i] == "X" and i % 14 != 13 and random.randint(0, 99) < opt.probs:
                    perturbedLevel.append(random.choice(pieces))
                else:
                    perturbedLevel.append(levelByColumn[i])
            testing_data.append((perturbedLevel, levelByColumn))
    return training_data, testing_data


training_data, testing_data = prepareData()
for i, data in enumerate(training_data):
    swapi = random.randrange(i, len(training_data))
    training_data[i], training_data[swapi] = training_data[swapi], data
for i, data in enumerate(testing_data):
    swapi = random.randrange(i, len(testing_data))
    testing_data[i], testing_data[swapi] = testing_data[swapi], data

EMBEDDING_DIM = opt.edim
HIDDEN_DIM = opt.hdim


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):

        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(tileMapping),
                   len(tileMapping))
model.cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in tqdm(range(opt.niter)):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network
        sentence_in = prepare_sequence(sentence, tileMapping)
        targets = prepare_sequence(tags, tileMapping)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "lstm_" + str(opt.tsize) + "_" + str(opt.niter) + ".pth")

# Find training accuracy
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for i in range(len(training_data)):
#         inputs = prepare_sequence(training_data[i][0], tileMapping)
#         tag_scores = model(inputs)
#         values, indices = torch.max(tag_scores, 1)
#         for j in range(len(indices)):
#             total += 1
#             if indices[j].item() == tileMapping[training_data[i][1][j]]:
#                 correct += 1
#     accuracy = (correct / total) * 100
#     print("Training accuracy " + str(round(accuracy, 2)) + "%")

# Find Test accuracy
with torch.no_grad():
    correct = 0
    fullCorrect = 0
    total = 0
    tilesIncorrect = [[0 for i in range(10)] for j in range(10)]
    for i in range(len(testing_data)):
        inputs = prepare_sequence(testing_data[i][0], tileMapping)
        tag_scores = model(inputs)
        values, indices = torch.max(tag_scores, 1)
        isCorrect = True
        for j in range(len(indices)):
            total += 1
            if indices[j].item() == tileMapping[testing_data[i][1][j]]:
                correct += 1
            else:
                isCorrect = False
                tilesIncorrect[tileMapping[testing_data[i][1][j]]][indices[j].item()] += 1
        if isCorrect:
            fullCorrect += 1
    accuracy = (correct / total) * 100
    print("Testing accuracy " + str(round(accuracy, 2)) + "%")
    fullAccuracy = (fullCorrect / len(testing_data)) * 100
    print("Testing levels that were fully correct: " +
          str(round(fullAccuracy, 2)) + "%")
    for row in tilesIncorrect:
        print(row)
