import torch
import torch.nn as nn
import argparse
import numpy as np
from config import config_mgr
from common import level
from seq2seq.models.lstm_tagger import LSTMTagger


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    res = torch.tensor(idxs, dtype=torch.long).cuda()
    return res


tileMapping = {"X": 0, "S": 1, "-": 2, "?": 3, "Q": 4,
               "E": 5, "<": 6, ">": 7, "[": 8, "]": 9}
piecesFull = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]"]
pipe = ["<", ">", "[", "]"]
revTileMapping = {v: k for k, v in tileMapping.items()}

DEF_LSTM_MODEL_FILE = 'lstm_300_100.pth'
DEF_EMBEDDING_DIM = 256
DEF_HIDDEN_DIM = 256

lstm = None


def load_lstm(model_file=DEF_LSTM_MODEL_FILE,
              embedding_dim=DEF_EMBEDDING_DIM,
              hidden_dim=DEF_HIDDEN_DIM,
              force_reload=False):
    global lstm

    if lstm is None or force_reload:
        assert torch.cuda.is_available()
        torch.cuda.set_device(0)

        lstm = LSTMTagger(embedding_dim, hidden_dim, len(tileMapping),
                          len(tileMapping))
        lstm.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            lstm = nn.DataParallel(lstm)

        model_path = config_mgr.get_absolute_path('seq2seq/trained_nets/' +
                                                  model_file)
        lstm.load_state_dict(torch.load(config_mgr
                                        .get_absolute_path(model_path)))
        lstm.eval()


def apply_lstm(level_as_text):
    level_by_cols = []
    level_by_rows = [list(line) for line in level_as_text.split('\n')]
    while level_by_rows and not level_by_rows[-1]:
        level_by_rows.pop()
    # We read in a text file version the level into a 2-D array
    # However, the LSTM will read the level column by column
    # So, it is necessary to swap rows and columns and then flatten the level
    for j in range(len(level_by_rows[0])):
        for i in range(len(level_by_rows)):
            level_by_cols.append(level_by_rows[i][j])

    with torch.no_grad():
        inputs = prepare_sequence(level_by_cols, tileMapping)
        tag_scores = lstm(inputs)
        values, indices = torch.max(tag_scores, 1)
        fixed_level = []
        for j in range(len(indices)):
            fixed_level.append(indices[j].item())
        fixed_level = np.reshape(np.array(fixed_level), (-1, 14))
        fixed_level = fixed_level.transpose((1, 0))

    return level.Level(data=fixed_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('level')
    parser.add_argument('--edim', type=int, default=256,
                        help='number of embedding dimensions')
    parser.add_argument('--hdim', type=int, default=256,
                        help='number of hidden dimensions')
    opt = parser.parse_args()
