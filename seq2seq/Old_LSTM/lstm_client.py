import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np

from config import config_mgr
from common import level
from models import lstm_tagger

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    res = torch.tensor(idxs, dtype=torch.long).cuda()
    return res


tileMapping = {"X": 0, "S": 1, "-": 2, "?": 3, "Q": 4,
               "E": 5, "<": 6, ">": 7, "[": 8, "]": 9}
piecesFull = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]"]
pipe = ["<", ">", "[", "]"]
revTileMapping = {v: k for k, v in tileMapping.items()}

DEF_LSTM_MODEL_FILE = 'trained_nets/lstm_200_175.pth'
DEF_EMBEDDING_DIM = 256
DEF_HIDDEN_DIM = 256

lstm = None

def load_lstm(model_file = DEF_LSTM_MODEL_FILE,
              embedding_dim = DEF_EMBEDDING_DIM,
              hidden_dim = DEF_HIDDEN_DIM,
              force_reload = False):
    global lstm

    if lstm is None or force_reload:
        assert torch.cuda.is_available()
        torch.cuda.set_device(0)

        model = LSTMTagger(embedding_dim, hidden_dim, len(tileMapping), len(tileMapping))
        model.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model_path = config_mgr.get_absolute_path('lstm/trained_nets/' + model_file))
        model.load_state_dict(torch.load(config_mgr.get_absolute_path(model_file)))
        model.eval()

def apply_lstm(level):
    if type(level) is not level.Level:
        raise TypeError("function apply_lstm() expects a Level class.")

    height, width = len(level_by_rows), len(level_by_rows[0])
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
        fixed_level = []
        for j in tqdm(range(len(indices))):
            fixed_level.append(revTileMapping[indices[j].item()])
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
