import torch
import numpy as np
import LoadData
#//////////////////////////////
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-uncased')
#加载预训练模型
pretrained = BertModel.from_pretrained("bert-base-uncased").to(device)

#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

class ExtractTextFeature(torch.nn.Module):
    def __init__(self,text_length,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # 256
        self.text_length = text_length  # 75
        self.fc = torch.nn.Linear(768, hidden_size*2)

    def forward(self,text):
        data = token.batch_encode_plus(batch_text_or_text_pairs=text,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=75,
                                       return_tensors='pt',
                                       return_length=True)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            #out.last_hidden_state：[32, 75, 768]      last_hidden_state[:, 0]:  torch.Size([32, 768])      out[0] [32,75,768]    out[1] [32, 768]
        #result = self.fc(out.last_hidden_state[:, 0])  #[32, 512]  X[:,0]就是取所有行的第0个数据
        result=self.fc(out[1])
        # out[0]：[512]
        seq = self.fc(out[0])

        return result,seq

if __name__ == "__main__":
    model = ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN).to(device)
    for text, text_index,image_feature,attribute_index,group,id in LoadData.train_loader:

        out,seq = model(text)
        print(out.shape) #[32, 512]
        print(seq.shape) #[32, 75, 512]

        break
