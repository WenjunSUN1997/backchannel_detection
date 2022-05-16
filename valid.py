import pandas as pd
import torch
import sys
import model_1
import data_class
import create_dataset
import model_2

batch_size = 8

if __name__ =="__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    len_block = int(sys.argv[1])
    model_path = 'judge_model_11_5'+sys.argv[1]
    model_test = model_1.back_channel_sentence_1(len_block, batch_size)
    model_test.load_state_dict(torch.load(model_path))
    model_test.to(device)

    with open('judge_model_11_5_'+str(2*len_block)+'test.txt', 'w') as file:
        file.write(' ')

    data_test = pd.read_csv('data_processed_10_5/data_test_'+sys.argv[1]+'.csv')
    data_test = data_class.data_loader(data_test)
    print(len(data_test))
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

    model_1.valid_model(model_test, test_loader)




