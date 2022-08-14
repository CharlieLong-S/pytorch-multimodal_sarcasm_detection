import torch
import numpy as np
import matplotlib.pyplot as plt

import LoadData


class ExtractAttributeFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractAttributeFeature, self).__init__()
        embedding_weight=self.getEmbedding()  #一个二维张量，每行上的参数
        self.embedding_size=embedding_weight.shape[1]   #每行上的参数数量
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)  #在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化，
        # embedding size 200->100
        """Raw attribute vectors e(ai) are passed through a two-layer neural network to obtain the attention weights αi for constructing the attribute guidance vector vattr."""
        self.Linear_1 = torch.nn.Linear(self.embedding_size, int(self.embedding_size/2))
        # embedding size 100->1
        self.Linear_2 = torch.nn.Linear(int(self.embedding_size/2),1)

    def forward(self, input):
        """
        e(a_i)
        """
        # -1 represent batch size ,在torch里面，view函数相当于numpy的reshape，  (batch size,5,embedding_size)
        self.embedded=self.embedding(input).view(-1, 5, self.embedding_size)
        """
        a_i=W_2*tanh(W_1*e(a_i)+b_1)+b_2
        """
        attn_weights = self.Linear_1(self.embedded.view(-1,self.embedding_size))
        # attn_weights = torch.nn.functional.tanh(attn_weights)
        attn_weights = torch.nn.functional.relu(attn_weights)
        attn_weights = self.Linear_2(attn_weights)
        """
        a=softmax(a) 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1
        dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。默认dim的方法已经弃用了，最好声明dim，否则会警告
        """
        attn_weights = torch.nn.functional.softmax(attn_weights.view(-1,5),dim=1)
        finalState = torch.bmm(attn_weights.unsqueeze(1), self.embedded).view(-1,200)   #计算两个tensor的矩阵乘法，torch.bmm(a,b),
        return finalState,self.embedded

    def getEmbedding(self):   #torch. from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        # numpy.loadtxt() 函数从文本文件中加载数据,返回值为从 txt 文件中读取的 N 维数组。
        return torch.from_numpy(np.loadtxt("multilabel_database_embedding/vector.txt", delimiter=' ', dtype='float32'))

if __name__ == "__main__":
    test=ExtractAttributeFeature()
    for text,text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        print(LoadData.train_loader.shape)
        #attribute_index [32, 5]
        result,seq=test(attribute_index)
        # [32, 200]
        print(result.shape)
        # [32, 5, 200]
        print(seq.shape)
        break


