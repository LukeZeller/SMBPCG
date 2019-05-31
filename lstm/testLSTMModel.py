import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from tqdm import tqdm
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('directory')
parser.add_argument('--edim', type=int, default=256,
                    help='number of embedding dimensions')
parser.add_argument('--hdim', type=int, default=256,
                    help='number of hidden dimensions')
parser.add_argument('--probp', type=int, default=60,
                    help='prob a pipe tile is changed')
parser.add_argument('--probs', type=int, default=10,
                    help='prob a stair tile is changed')
parser.add_argument('--tsize', type=int, default=100,
                    help='number of training sets create per level')
opt = parser.parse_args()

assert torch.cuda.is_available()
torch.cuda.set_device(0)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    res = torch.tensor(idxs, dtype=torch.long).cuda()
    return res


tileMapping = {"X": 0, "S": 1, "-": 2, "?": 3, "Q": 4,
               "E": 5, "<": 6, ">": 7, "[": 8, "]": 9}
piecesFull = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]"]
pipe = ["<", ">", "[", "]"]
revTileMapping = {v: k for k, v in tileMapping.items()}


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
model.load_state_dict(torch.load("./" + opt.file))
model.eval()


# probs = probability of a stair rock forming due to an adjacent stair block.
# probg = probability of a stair rock forming due to an adjacent ground block.
# Note: Here probability is in [0, 1] not w.r.t 100 (unlike below)
def _add_some_rocks(level_by_col, probs, probg):
    sz = len(level_by_col)
    for i in range(sz):
        # Ignore the ground level
        if i % 14 == 13 or level_by_col[i] == 'X':
            continue
        # n_adj_s = # of direct neighbors that are stair blocks
        n_adj_s = 0
        # offset of all neighbors excluding neighbor below, which
        # is handled separately to account for ground tiles
        for offset in [-14, -1, 14]:
            j = i + offset
            if j >= 0 and j < sz and level_by_col[j] == 'X':
                n_adj_s += 1
        # n_adj_g = # of direct neighbors that are ground blocks and is always in {0, 1}
        # Note - overflow here should be impossible as the length of level_by_cols
        #        is a multiple of the height ( = 14)
        n_adj_g = (i % 14 == 12 and level_by_col[i + 1] == 'X')
        # Why do I model formation this way? Absolutely no reason.
        p_form = 1 - (1 - probs) ** n_adj_s * (1 - probg) ** n_adj_g
        if random.uniform(0, 1) < p_form:
            level_by_col[i] = 'X'

# probp_d == probability of pipe tile being deleted
# probs_d == probability of stair tile (== ground tile
#            above ground) being deleted.
# probs_c == probability of tile near stair tile being 'created'
#            (i.e. transformed into a stair tile)


def _perturb_level(level_by_col, probp_d, probs_d, probs_c):
    perturbed_level = []
    level_cpy = level_by_col[:]
    for i in range(len(level_cpy)):
        # Initial perturbations: Delete pipe and stair tiles.
        # If first pipe tile that will be encountered by the LSTM
        # is deleted, the rest of the pipe must be as well for otherwise
        # the poor lil' network stands no chance
        if level_cpy[i] in pipe and random.randint(0, 99) < probp_d:
            perturbed_level.append(random.choice(piecesFull))
            if level_cpy[i] == "<":
                p_itr = i
                while level_cpy[p_itr] in pipe:
                    level_cpy[p_itr] = "-"
                    level_cpy[p_itr + 14] = "-"
                    p_itr += 1
        elif level_cpy[i] == "X" and i % 14 != 13 and random.randint(0, 99) < probs_d:
                perturbed_level.append(random.choice(piecesFull))
        else:
            perturbed_level.append(level_cpy[i])

    probs_c /= 100
    _add_some_rocks(perturbed_level, probs_c, probs_c / 2)
    _add_some_rocks(perturbed_level, probs_c, probs_c / 2)

    return (perturbed_level, level_cpy)


def prepare_data():
    training_data = []
    testing_data = []
    for filename in os.listdir(opt.directory):
        level_by_cols = []
        with open(opt.directory + "/" + filename) as text_f:
            level_by_rows = np.array([list(line) for line in text_f])
        level_by_rows = level_by_rows[:, :-1]
        # We read in a text file version the level into a 2-D array
        # However, the LSTM will read the level column by column
        # So, it is necessary to swap rows and columns and then flatten the level
        for j in range(len(level_by_rows[0])):
            for i in range(len(level_by_rows)):
                level_by_cols.append(level_by_rows[i][j])
        # Create Training Data
        for k in range(opt.tsize):
            # TODO: Add option for distinct staircase creation probability
            training_data.append(_perturb_level(level_by_cols, opt.probp, opt.probs, opt.probs))
        for k in range(int(opt.tsize / 10) + 1):
            testing_data.append(_perturb_level(level_by_cols, opt.probp, opt.probs, opt.probs))
    return training_data, testing_data


training_data, testing_data = prepare_data()


with torch.no_grad():
    correct = 0
    fullCorrect = 0
    total = 0
    tilesIncorrect = [[0 for i in range(10)] for j in range(10)]
    for i in tqdm(range(len(testing_data))):
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


with torch.no_grad():
    inputs = prepare_sequence(testing_data[0][0], tileMapping)
    tag_scores = model(inputs)
    values, indices = torch.max(tag_scores, 1)
    tempLevel = []
    pertLevel = []
    realLevel = []
    for j in tqdm(range(len(indices))):
        tempLevel.append(revTileMapping[indices[j].item()])
        pertLevel.append(testing_data[0][0][j])
        realLevel.append(testing_data[0][1][j])
    tempLevel = np.reshape(np.array(tempLevel), (-1, 14))
    tempLevel = tempLevel.transpose((1, 0))
    pertLevel = np.reshape(np.array(pertLevel), (-1, 14))
    pertLevel = pertLevel.transpose((1, 0))
    realLevel = np.reshape(np.array(pertLevel), (-1, 14))
    realLevel = pertLevel.transpose((1, 0))
    fixedLevel = ""
    perturbedLevel = ""
    rLevel = ""
    for i in range(len(tempLevel)):
        for j in range(len(tempLevel[0])):
            fixedLevel += tempLevel[i][j]
            perturbedLevel += pertLevel[i][j]
            rLevel += realLevel[i][j]
        fixedLevel += "\n"
        perturbedLevel += "\n"
        rLevel += "\n"
    file = open("fixed_level.txt", 'w')
    file1 = open("perturbed_level.txt", 'w')
    file2 = open("og_evel.txt", 'w')
    file.write(fixedLevel)
    file1.write(perturbedLevel)
    file2.write(rLevel)
    file1.close()
    file.close()
    file2.close()
