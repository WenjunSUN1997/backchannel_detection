import pandas as pd
import torch
import create_dataset
import data_class
import model_1
import model_2
import model_3
import sys

batch_size = 8

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    len_block = int(sys.argv[1])
    with open('judge_result_13_5_'+str(2*len_block)+'.txt', 'w') as file:
        file.write(' ')

    data_train = pd.read_csv('data_processed_10_5/data_train_'+sys.argv[1]+'.csv')
    data_train = data_train.reset_index(drop=True)
    data_train = data_class.data_loader(data_train)
    print(data_train.__len__())
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    model = model_3.back_channel_sentence_1(len_block, batch_size)
    model.to(device)
    model_3.train_model(model, train_loader)

    torch.save(model.state_dict(), 'judge_model_13_5'+sys.argv[1])

