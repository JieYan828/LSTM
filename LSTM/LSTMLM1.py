import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number_dict[
                word[word_index + n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch  # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.Emb = nn.Embedding(n_class, embedding_dim=emb_size)

        self.F = nn.Linear(n_hidden + emb_size, n_hidden, bias=True)
        self.I = nn.Linear(n_hidden + emb_size, n_hidden, bias=True)
        self.C = nn.Linear(n_hidden + emb_size, n_hidden, bias=True)
        self.O = nn.Linear(n_hidden + emb_size, n_hidden, bias=True)

        self.out = nn.Linear(n_hidden, n_class, bias=True)

    def forward(self, X):
        X = self.Emb(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]

        c = [] #??????????????????
        h = [] #????????????????????????
        c.append(torch.zeros(batch_size, n_hidden).to(device))
        h.append(torch.zeros(batch_size, n_hidden).to(device))
        # ??????5???????????????
        for i in range(n_step):
            # ?????????
            h_x_0 = torch.cat((h[i], X[i]), 1).to(device)
            # ?????????
            f_0_ = torch.sigmoid(self.F(h_x_0))
            # ?????????
            i_0_ = torch.sigmoid(self.I(h_x_0))
            c_0_ = torch.tanh(self.C(h_x_0))
            # ????????????
            c_1 = f_0_.mul(c[i]) + i_0_.mul(c_0_)
            c.append(c_1)
            # ?????????
            o_0 = torch.sigmoid(self.O(h_x_0))
            h_1 = o_0.mul(torch.tanh(c_1))

            h.append(h_1)

        # # ??????
        # h_x_0 = torch.cat((torch.zeros(batch_size, n_hidden), X[0]), 1)
        # # ?????????
        # f_0_ = self.sigma_h(self.F(h_x_0))
        # # ?????????
        # i_0_ = self.sigma_i(self.I(h_x_0))
        # c_0_ = self.tanh_c(self.C(h_x_0))
        # # ????????????
        # c_1 = torch.mul(f_0_, torch.zeros(batch_size, n_hidden)) + torch.mul(i_0_, c_0_)
        # # ?????????
        # o_1 = self.sigma_o(self.O(h_x_0))
        # h_1 = torch.mul(o_1, self.tanh_o(c_1))
        #
        # # ??????
        # h_x_1 = torch.cat((h_1, X[1]), 1)
        # # ?????????
        # f_1_ = self.sigma_h(self.F(h_x_1))
        # # ?????????
        # i_1_ = self.sigma_i(self.I(h_x_1))
        # c_1_ = self.tanh_c(self.C(h_x_1))
        # # ????????????
        # c_2 = torch.mul(f_1_, c_1) + torch.mul(i_1_, c_1_)
        # # ?????????
        # o_1 = self.sigma_o(self.O(h_x_1))
        # h_2 = torch.mul(o_1, self.tanh_o(c_2))
        #
        # # ??????
        # h_x_2 = torch.cat((h_2, X[2]), 1)
        # # ?????????
        # f_2_ = self.sigma_h(self.F(h_x_2))
        # # ?????????
        # i_2_ = self.sigma_i(self.I(h_x_2))
        # c_2_ = self.tanh_c(self.C(h_x_2))
        # # ????????????
        # c_3 = torch.mul(f_2_, c_2) + torch.mul(i_2_, c_2_)
        # # ?????????
        # o_2 = self.sigma_o(self.O(h_x_2))
        # h_3 = torch.mul(o_2, self.tanh_o(c_3))
        #
        # # ??????
        # h_x_3 = torch.cat((h_3, X[3]), 1)
        # # ?????????
        # f_3_ = self.sigma_h(self.F(h_x_3))
        # # ?????????
        # i_3_ = self.sigma_i(self.I(h_x_3))
        # c_3_ = self.tanh_c(self.C(h_x_3))
        # # ????????????
        # c_4 = torch.mul(f_3_, c_3) + torch.mul(i_3_, c_3_)
        # # ?????????
        # o_3 = self.sigma_o(self.O(h_x_3))
        # h_4 = torch.mul(o_3, self.tanh_o(c_4))
        #
        # # ??????
        # h_x_4 = torch.cat((h_4, X[4]), 1)
        # # ?????????
        # f_4_ = self.sigma_h(self.F(h_x_4))
        # # ?????????
        # i_4_ = self.sigma_i(self.I(h_x_4))
        # c_4_ = self.tanh_c(self.C(h_x_4))
        # # ????????????
        # c_5 = torch.mul(f_4_, c_4) + torch.mul(i_4_, c_4_)
        # # ?????????
        # o_4 = self.sigma_o(self.O(h_x_4))
        # h_5 = torch.mul(o_4, self.tanh_o(c_5))


        output = torch.stack((h[1], h[2], h[3], h[4], h[5]), 0)
        output = output[-1]
        model = self.out(output)
        return model


def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch + 1}.ckpt')


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}????????????????????????")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell
    batch_size = 128  # batch size
    learn_rate = 0.0005
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embeding size
    save_checkpoint_epoch = 5  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt')  # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    print("\nTrain the LSTMLM????????????????????????")
    train_LSTMlm()

    print("\nTest the LSTMLM????????????????????????")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)
