import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import os

random.seed(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('--niter', type=int, default=10,
                    help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--hdim', type=int, default=256,
                    help='number of hidden dimensions')
parser.add_argument('--tsize', type=int, default=10,
                    help='number of training sets create per level')
parser.add_argument('--prob', type=int, default=30,
                    help='prob a pipe tile is changed')
opt = parser.parse_args()

assert torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ["<", ">", "[", "]"]
pieces = ["X", "S", "-", "?", "Q", "E", "<", ">", "[", "]"]
tileMapping = {"X": 0, "S": 1, "-": 2, "?": 3, "Q": 4,
               "E": 5, "<": 6, ">": 7, "[": 8, "]": 9,
               "`": 10}
revTileMapping = {v: k for k, v in tileMapping.items()}
SOS_token = 10


def prepare_sequence(seq, to_index):
    indexes = [to_index[w] for w in seq]
    res = torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    return res


def tensorsFromPair(pair):
    input_tensor = prepare_sequence(pair[0], tileMapping)
    target_tensor = prepare_sequence(pair[1], tileMapping)
    return (input_tensor, target_tensor)


def prepareData():
    levelByColumnArray = []
    training_data = []
    testing_data = []
    fileIndex = 0
    for filename in os.listdir(opt.directory):
        with open(opt.directory + "/" + filename) as textFile:
            levelByRow = [list(line) for line in textFile]

        # We read in a text file version the level into a 2-D array
        # However, the LSTM will read the level column by column
        # So, it is necessary to swap rows and columns

        levelByColumn = ""
        for j in range(len(levelByRow[0])):
            for i in range(len(levelByRow)):
                levelByColumn += str(levelByRow[i][j]) + " "
        levelByColumnArray.append(levelByColumn.split())
        training_data.append((levelByColumnArray[fileIndex], levelByColumnArray[fileIndex]))
        testing_data.append((levelByColumnArray[fileIndex], levelByColumnArray[fileIndex]))

        # Create Training Data
        for k in range(opt.tsize):
            proturbedLevel = ""
            for i in range(len(levelByColumnArray[fileIndex])):
                if levelByColumnArray[fileIndex][i] in pipe and random.randint(0, 99) < opt.prob:
                    proturbedLevel += random.choice(pieces) + " "
                else:
                    proturbedLevel += levelByColumnArray[fileIndex][i] + " "
            training_data.append((proturbedLevel.split(), levelByColumnArray[fileIndex]))

        # Create Testing Data
        for k in range(opt.tsize):
            proturbedLevel = ""
            for i in range(len(levelByColumnArray[fileIndex])):
                if levelByColumnArray[fileIndex][i] in pipe and random.randint(0, 99) < opt.prob:
                    proturbedLevel += random.choice(pieces) + " "
                else:
                    proturbedLevel += levelByColumnArray[fileIndex][i] + " "
            testing_data.append((proturbedLevel.split(), levelByColumnArray[fileIndex]))
        fileIndex += 1
    return training_data, testing_data, levelByColumnArray


training_data, testing_data, levelByColumnArray = prepareData()


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
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max([len(training_data[i][0]) for i in range(len(training_data))]))
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


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(target_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
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

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, opt.niter + 1)):
        for i in range(len(training_data)):
            training_pair = tensorsFromPair(training_data[i])
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

        train(input_tensor, target_tensor, encoder,
              decoder, encoder_optimizer, decoder_optimizer, criterion)


def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = prepare_sequence(sentence, tileMapping)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(len(sentence), encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(len(sentence), len(sentence), device=device)

        for di in range(len(sentence)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(revTileMapping[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


hidden_size = opt.hdim
encoder1 = EncoderRNN(len(tileMapping), hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, len(tileMapping), dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1)

torch.save({'encoder': encoder1.state_dict(), 'decoder': attn_decoder1.state_dict}, "lstm_" + str(opt.tsize) + "_" + str(opt.niter) + ".pth")

trainingCorrect = 0
trainingFullyCorrect = 0
total = 0
for i in tqdm(range(len(training_data))):

    decoded_words_train, decoder_attentions = evaluate(encoder1, attn_decoder1, training_data[i][0])
    isCorrect = True
    for j in range(len(decoded_words_train) - 1):
        total += 1
        # print("Training Data " + str(training_data[i][1][j]))
        # print("Decoded Word " + str(decoded_words_train[j]))
        if decoded_words_train[j] == training_data[i][1][j]:
            trainingCorrect += 1
        else:
            isCorrect = False
    if isCorrect:
        trainingFullyCorrect += 1

accuracy = (trainingCorrect / total) * 100
print("Training accuracy " + str(round(accuracy, 2)) + "%")
print("Training level fully correct accuracy " + str(round(trainingFullyCorrect / len(training_data), 2)) + "%")

testingCorrect = 0
testingFullyCorrect = 0
total = 0
for i in tqdm(range(len(testing_data))):
    decoded_words_test, decoder_attentions = evaluate(encoder1, attn_decoder1, testing_data[i][0])
    isCorrect = True
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
