import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('level')
parser.add_argument('--edim', type=int, default=256,
                    help='number of embedding dimensions')
parser.add_argument('--hdim', type=int, default=256,
                    help='number of hidden dimensions')
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

level_by_cols = []
with open("./" + opt.level) as text_f:
        level_by_rows = np.array([list(line) for line in text_f])
        level_by_rows = level_by_rows[:, :-1]
        # We read in a text file version the level into a 2-D array
        # However, the LSTM will read the level column by column
        # So, it is necessary to swap rows and columns and then flatten the level
        for j in range(len(level_by_rows[0])):
            for i in range(len(level_by_rows)):
                level_by_cols.append(level_by_rows[i][j])

with torch.no_grad():
    inputs = prepare_sequence(level_by_cols, tileMapping)
    tag_scores = model(inputs)
    values, indices = torch.max(tag_scores, 1)
    tempLevel = []
    for j in tqdm(range(len(indices))):
        tempLevel.append(revTileMapping[indices[j].item()])
    tempLevel = np.reshape(np.array(tempLevel), (-1, 14))
    tempLevel = tempLevel.transpose((1, 0))
    fixedLevel = ""
    for i in range(len(tempLevel)):
        for j in range(len(tempLevel[0])):
            fixedLevel += tempLevel[i][j]
        fixedLevel += "\n"
    file = open("fixed_" + opt.level, 'w')
    file.write(fixedLevel)
    file.close()
