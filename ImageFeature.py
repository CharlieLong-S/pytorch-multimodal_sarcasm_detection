import torch
import numpy as np
import matplotlib.pyplot as plt

import LoadData


class ExtractImageFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractImageFeature, self).__init__()
        # 2048->1024
        self.Linear = torch.nn.Linear(1000, 500)

    def forward(self, input):
        #图片img的size比如是（28，28，3）就可以利用img.permute(2,0,1)得到一个size为（3，28，28）的tensor。
        #(196,32,1000)
        input=input.permute(1,0,2)
        output=list()
        for i in range(196):  #使用一个预训练和微调的ResNet模型来获得图片的14*14区域向量，然后将原始向量平均
            #(32,500)
            sub_output=torch.nn.functional.relu(self.Linear(input[i]))
            output.append(sub_output)
        #(196,32, 500)
        output=torch.stack(output) #dim默认为08
        # (32,500)
        mean=torch.mean(output,0)  #Returns the mean value of each row of the input tensor in the given dimension dim
        return mean,output

if __name__ == "__main__":
    test=ExtractImageFeature() #module子类
    for text,text_index,image_feature,attribute_index,group,id in LoadData.train_loader: #字典中的键
        print(len(LoadData.train_loader))
        #image_feature  (32, 196, 1000)
        result,seq=test(image_feature)
        # [32, 500]
        print(result.shape)
        # [196, 32, 500]
        print(seq.shape)
        break


