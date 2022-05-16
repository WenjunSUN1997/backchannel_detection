import os
import pandas as pd

def get_data(len_block:int):
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

    for col_name in list(data_main):
        if str(len_block) in col_name:
            data_main = data_main[['sentence',col_name]]
            break

    for col_name in list(data_back):
        if str(len_block) in col_name:
            data_back= data_back[['sentence',col_name]]
            break
    data_main = data_main.dropna()
    data_back = data_back.dropna()
    if len(data_main)<len(data_back):
        max_len = len(data_main)
    else:
        max_len = len(data_back)
    data_main = pd.DataFrame({'sentence': data_main['sentence'],
                              'block': data_main['block_'+str(len_block)],
                              'target': [0] * len(data_main)})[:max_len]
    data_back = pd.DataFrame({'sentence': data_back['sentence'],
                              'block': data_back['block_'+str(len_block)],
                              'target': [1] * len(data_back)})[:max_len]
    print('done')
    return data_main, data_back
