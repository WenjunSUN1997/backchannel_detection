import os
import pandas as pd

def get_data():
    file_list = os.listdir('data_processed_10_5')
    file_main = [x for x in file_list if 'main_block' in x]
    file_back = [x for x in file_list if 'back_block' in x]

    for index in range(len(file_main)):
        if index == 0:
            data_main = pd.read_csv('data_processed_10_5/'+file_main[index])
        else:
            data_main = data_main.append(pd.read_csv('data_processed_10_5/'+file_main[index]), ignore_index=True)

    for index in range(len(file_back)):
        if index == 0:
            data_back = pd.read_csv('data_processed_10_5/' + file_back[index])
        else:
            data_back = data_back.append(pd.read_csv('data_processed_10_5/' + file_back[index]), ignore_index=True)


    data_main = data_main.dropna()
    data_back = data_back.dropna()
    if len(data_main)<len(data_back):
        max_len = len(data_main)
    else:
        max_len = len(data_back)

    data_main = data_main[:max_len]
    data_back = data_back[:max_len]
    data_main['target'] = [0] * len(data_main)
    data_back['target'] = [1] * len(data_back)
    data_main.reset_index(drop=True)
    data_back.reset_index(drop=True)
    print('done')
    return data_main, data_back
