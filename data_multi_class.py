import pandas as pd
from transformers import AutoTokenizer, BertModel
from torch import tensor
from torch.utils.data import Dataset
from ast import literal_eval
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class data_loader(Dataset):
    def __init__(self,data):
        self.data = data
        self.max_len_sentence = 6
        self.max_len_block = 256
        self.tokenizer = AutoTokenizer.from_pretrained('camembert-base')
        self.bert_model = BertModel.from_pretrained('camembert-base')

    def __len__(self):
        return len(self.data['sentence'])

    def __getitem__(self, item):
        print(item)
        sentence = self.data['sentence'][item]
        block_list_1 = literal_eval(self.data['block_1'][item])
        block_list_2 = literal_eval(self.data['block_2'][item])
        block_list_3 = literal_eval(self.data['block_3'][item])
        if len(sentence) >= self.max_len_sentence:
            sentence = sentence[:self.max_len_sentence]
        for block_list in [block_list_1, block_list_2, block_list_3]:
            for index in range(len(block_list)):
                if len(block_list[index]) >= self.max_len_block:
                    block_list[index] = block_list[index][:self.max_len_block]
        target = self.data['target'][item]
        print(sentence)
        output_tokenizer = self.tokenizer(block_list_1, max_length=self.max_len_block, padding='max_length')
        input_ids = tensor(output_tokenizer['input_ids'])
        attention_mask = tensor(output_tokenizer['attention_mask'])
        block_1_bert_feature = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output_tokenizer = self.tokenizer(block_list_2, max_length=self.max_len_block, padding='max_length')
        input_ids = tensor(output_tokenizer['input_ids'])
        attention_mask = tensor(output_tokenizer['attention_mask'])
        block_2_bert_feature = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output_tokenizer = self.tokenizer(block_list_3, max_length=self.max_len_block, padding='max_length')
        input_ids = tensor(output_tokenizer['input_ids'])
        attention_mask = tensor(output_tokenizer['attention_mask'])
        block_3_bert_feature = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output_tokenizer = self.tokenizer([sentence], max_length=self.max_len_sentence, padding='max_length')
        input_ids = tensor(output_tokenizer['input_ids'])
        attention_mask = tensor(output_tokenizer['attention_mask'])
        sentence_bert_feature = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return {'sentence_bert_feature': sentence_bert_feature['last_hidden_state'][0][0].to(device),
                'block_1_bert_feature': block_1_bert_feature['last_hidden_state'][:, :1, :].squeeze().to(device),
                'block_2_bert_feature': block_2_bert_feature['last_hidden_state'][:, :1, :].squeeze().to(device),
                'block_3_bert_feature': block_3_bert_feature['last_hidden_state'][:, :1, :].squeeze().to(device),
                'target': tensor(target).to(device)}



