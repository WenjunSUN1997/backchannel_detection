import pandas as pd
from ast import literal_eval
from transformers import  AutoTokenizer, BertModel
import torch
import numpy as np
import gc

import create_multi_dataset
import data_class
from sklearn.utils import shuffle

import os
import create_dataset
from torch import tensor

a = torch.randn(2,2,1)
print(a)
a= a.reshape(2,1,2)
print(a)


# data_main, data_back = create_multi_dataset.get_data()
#
# data_all = data_main.append(data_back)
# data_all = shuffle(data_all).reset_index(drop=True)
# print(data_all)
# data_all_train = data_all[:int(0.7*len(data_all))]
# data_all_test = data_all[int(0.7*len(data_all)):]
# data_all_train.to_csv('data_processed_10_5/data_all_train.csv')
# data_all_test.to_csv('data_processed_10_5/data_all_test.csv')


#
# data_main, data_back = create_dataset.get_data(1)
# data_main = data_class.data_loader(data_main)
# data_main = torch.utils.data.DataLoader(data_main, batch_size=8, shuffle=True)
# for index, data in enumerate(data_main):
#     print(index)
#     print(data['sentence_bert_feature'].size())
#     print(data['block_bert_feature'].size())

# a = torch.randn(8,1,768)
# b = torch.randn(8,2,768)
#
# lstm_block = torch.nn.LSTM(
#     input_size=768,
#     hidden_size=128,
#     num_layers=2,
#     batch_first=True
# )
# lstm_sentence = torch.nn.LSTM(
#     input_size=768,
#     hidden_size=128,
#     num_layers=2,
#     batch_first=True
# )
#
# a = lstm_sentence(a)[0]
# b = lstm_block(b)[0]
# print(a.size())
# print(b.size())
#
# c = torch.cat((a, b),1)
# print(c.size())
# a = torch.randn(8,768)
# a =torch.unsqueeze(a, 1)
# print(a.size())
# data_main, data_back = create_dataset.get_data(3)
# data_main = shuffle(data_main).reset_index(drop=True)
# data_back = shuffle(data_back).reset_index(drop=True)
# print(len(data_main), len(data_back))
# data_1 = data_main.append(data_back)
# print(len(data_1))
# data_1 = shuffle(data_1).reset_index(drop=True)
# print(data_1)
# data_1_train = data_1[:int(0.7*len(data_1))]
# data_1_test = data_1[int(0.7*len(data_1)):]
# data_1_train.to_csv('data_processed_10_5/data_train_3.csv')
# data_1_test.to_csv('data_processed_10_5/data_test_3.csv')