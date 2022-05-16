import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score

class back_channel_sentence_1(torch.nn.Module):
    def __init__(self, len_block:int, batch_size:int):
        super(back_channel_sentence_1, self).__init__()
        self.len_block = 2*len_block
        self.batch_size = batch_size
        self.lstm_block = torch.nn.LSTM(
            input_size =  self.len_block*768,
            hidden_size = 768,
            num_layers = 2,
            batch_first= True
        )

        self.dropout = torch.nn.Dropout(0.2)
        self.active = torch.nn.ReLU()
        self.full_connection = torch.nn.Linear(2*768, 2)

    def forward(self, bert_feature_sentence, bert_feature_block):
        bert_feature_block = bert_feature_block.reshape(bert_feature_sentence.size(0), 1,self.len_block*768)
        x_block = self.lstm_block(bert_feature_block)
        x_block = x_block[0]
        print(x_block.size())

        x_sentence = torch.unsqueeze(bert_feature_sentence, 1)
        print(x_sentence.size())

        x_input = torch.cat((x_sentence, x_block), 1)
        print(x_input.size())

        # x_input = self.lstm_relation(x_input)
        # x_input =  x_input[0]
        # print(x_input.size())
        x_input = self.dropout(x_input)
        x_input = x_input.reshape(x_input.size(0), 2*768)
        x_input = self.active(x_input)
        out = self.full_connection(x_input)
        return out


def train_model(model:back_channel_sentence_1, data_loader):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=3)
    epoch = 10

    model.train()
    for num in range(epoch):
        result = []
        true = []
        amount_all = 0
        amount_correct = 0
        for index, real_data in enumerate(data_loader):
            bert_feature_block = real_data['block_bert_feature']
            bert_feature_sentence = real_data['sentence_bert_feature']
            target = real_data['target']
            out = model(bert_feature_sentence, bert_feature_block)
            loss = loss_function(out, target)
            max_value, out_index = torch.max(out, dim=1)
            result.append(out_index)
            true.append(target)
            n_correct = (out_index == target).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            amount_correct += n_correct
            amount_all += len(target)
            print('\033[1;32;46m' +str(n_correct / len(target)) +'\033[0m')
            print(n_correct / len(target))
            if index % 5 ==0:
                with open('judge_result_13_5_'+str(model.len_block)+'.txt', 'a+') as file:
                    file.write(str(amount_correct / amount_all) + '\n')
        print('\033[1;32;46m' +str(amount_correct / amount_all) +'\033[0m')
        print(amount_correct / amount_all)
        with open('judge_result_13_5_'+str(model.len_block)+'.txt', 'a+') as file:
            file.write('----------------------'+
                       str(amount_correct / amount_all) +
                       '-------------------'+'\n')
        torch.save(model.state_dict(), 'judge_model_13_5' + sys.argv[1])

def valid_model(model:back_channel_sentence_1, data_loader):
    model.eval()
    amount_all = 0
    amount_correct = 0
    for index, real_data in enumerate(data_loader):
        bert_feature_block = real_data['block_bert_feature']
        bert_feature_sentence = real_data['sentence_bert_feature']
        target = real_data['target']
        out = model(bert_feature_sentence, bert_feature_block)
        max_value, out_index = torch.max(out, dim=1)
        n_correct = (out_index == target).sum().item()
        amount_correct += n_correct
        amount_all += len(target)
        print(n_correct / len(target))
        with open('judge_model_12_5_'+str(model.len_block)+'test.txt', 'a+') as file:
            file.write(str(n_correct / len(target)) + '\n')
    print('\033[1;32;46m' + str(amount_correct / amount_all) + '\033[0m')
    print(amount_correct / amount_all)
    with open('judge_model_12_5_'+str(model.len_block)+'test.txt', 'a+') as file:
        file.write('----------------------' +
                   str(amount_correct / amount_all) +
                   '-------------------' + '\n')









