import pandas as pd
import torch
import sys
from model_1 import back_channel_sentence_1
import data_class
import create_dataset
import model_2
import data_multi_class

batch_size = 8



def valid_multi_model(model1:back_channel_sentence_1,
                      model2:back_channel_sentence_1,
                      model3:back_channel_sentence_1,
                      data_loader):
    model1.eval()
    model2.eval()
    model3.eval()
    amount_all = 0
    amount_correct = 0
    for index, real_data in enumerate(data_loader):
        bert_feature_block_1 = real_data['block_1_bert_feature']
        bert_feature_block_2 = real_data['block_2_bert_feature']
        bert_feature_block_3 = real_data['block_3_bert_feature']
        bert_feature_sentence = real_data['sentence_bert_feature']
        target = real_data['target']
        out_1 = model1(bert_feature_sentence, bert_feature_block_1)
        out_2 = model2(bert_feature_sentence, bert_feature_block_2)
        out_3 = model3(bert_feature_sentence, bert_feature_block_3)
        out = out_3+out_2+out_1
        max_value, out_index = torch.max(out, dim=1)
        n_correct = (out_index == target).sum().item()
        amount_correct += n_correct
        amount_all += len(target)
        print(n_correct / len(target))
        with open('result/multi_judge_model_11_5_test.txt', 'a+') as file:
            file.write(str(n_correct / len(target)) + '\n')
    print('\033[1;32;46m' + str(amount_correct / amount_all) + '\033[0m')
    print(amount_correct / amount_all)
    with open('result/multi_judge_model_11_5_test.txt', 'a+') as file:
        file.write('----------------------' +
                   str(amount_correct / amount_all) +
                   '-------------------' + '\n')

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model1 = back_channel_sentence_1(1, batch_size)
    model2 = back_channel_sentence_1(2, batch_size)
    model3 = back_channel_sentence_1(3, batch_size)

    model1.load_state_dict(torch.load('model/judge_model_11_51'))
    model2.load_state_dict(torch.load('model/judge_model_11_52'))
    model3.load_state_dict(torch.load('model/judge_model_11_53'))
    model1.to(device)
    model2.to(device)
    model3.to(device)

    data_test = pd.read_csv('data_processed_10_5/data_all_test.csv')
    data_test = data_multi_class.data_loader(data_test)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

    valid_multi_model(model1, model2, model3, test_loader)