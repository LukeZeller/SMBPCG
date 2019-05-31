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
parser.add_argument('--niter', type=int, default=100,
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ["<", ">", "[", "]"]
pieces = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]"]
tileMapping = {"X": 0, "S": 1, "-": 2, "?": 3, "Q": 4,
               "E": 5, "<": 6, ">": 7, "[": 8, "]": 9}
revTileMapping = {v: k for k, v in tileMapping.items()}


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
            if i % 14 == 0 and offset == -1:
                continue
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
            perturbed_level.append(random.choice(pieces))
            if level_cpy[i] == "<":
                p_itr = i
                while level_cpy[p_itr] in pipe:
                    level_cpy[p_itr] = "-"
                    level_cpy[p_itr + 14] = "-"
                    p_itr += 1
        elif level_cpy[i] == "X" and i % 14 != 13 and random.randint(0, 99) < probs_d:
                perturbed_level.append(random.choice(pieces))
        else:
            perturbed_level.append(level_cpy[i])

    probs_c /= 100
    _add_some_rocks(perturbed_level, probs_c, probs_c / 8)
    _add_some_rocks(perturbed_level, probs_c, probs_c / 8)

    return (perturbed_level, level_cpy)


def prepare_data():
    training_data = []
    testing_data = []
    maxL = 0
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
        maxL = max(maxL, len(level_by_cols))
        for k in range(opt.tsize):
            # TODO: Add option for distinct staircase creation probability
            training_data.append(_perturb_level(level_by_cols, opt.probp, opt.probs, opt.probs))
        for k in range(int(opt.tsize / 10) + 1):
            testing_data.append(_perturb_level(level_by_cols, opt.probp, opt.probs, opt.probs))
    return training_data, testing_data, maxL


training_data, testing_data, MAX_LENGTH = prepare_data()
for i, data in enumerate(training_data):
    swapi = random.randrange(i, len(training_data))
    training_data[i], training_data[swapi] = training_data[swapi], data
for i, data in enumerate(testing_data):
    swapi = random.randrange(i, len(testing_data))
    testing_data[i], testing_data[swapi] = testing_data[swapi], data


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def tensorFromSequence(seq, mapping):
    indexes = [mapping[w] for w in seq]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSequence(pair[0], tileMapping)
    target_tensor = tensorFromSequence(pair[1], tileMapping)
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(1, input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[2]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, learning_rate=0.01):
    n_iters = opt.niter

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in tqdm(range(n_iters)):
        for i in range(len(training_data)):
            training_pair = training_data[i]
            input_tensor = tensorFromSequence(training_pair[0], tileMapping)
            target_tensor = tensorFromSequence(training_pair[1], tileMapping)

        train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)


def evaluate(encoder, decoder, sequence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSequence(sequence, tileMapping)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(1, input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[2]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(revTileMapping[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


hidden_size = opt.hdim
encoder1 = EncoderRNN(len(training_data), hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, len(training_data)).to(device)

trainIters(encoder1, attn_decoder1)

torch.save({'encoder': encoder1.state_dict(), 'decoder': attn_decoder1.state_dict}, "lstm_" + str(opt.tsize) + "_" + str(opt.niter) + ".pth")

testingCorrect = 0
testingFullyCorrect = 0
total = 0
for i in tqdm(range(len(testing_data))):
    decoded_words_test, decoder_attentions = evaluate(encoder1, attn_decoder1, testing_data[i][0])
    isCorrect = True
    print(decoded_words_test)
    for j in range(len(decoded_words_test) - 1):
        total += 1
        if decoded_words_test[j] == testing_data[i][1][j]:
            testingCorrect += 1
        else:
            isCorrect = False
    if isCorrect:
        testingFullyCorrect += 1
accuracy = (testingCorrect / total) * 100
print("Testing accuracy " + str(round(accuracy, 2)) + "%")
print("Training level fully correct accuracy " + str(round(testingFullyCorrect / len(testing_data), 2)) + "%")
