
import torch
import numpy as np
import LoadData
import TextFeature
import AttributeFeature
import ImageFeature
import FuseAllFeature

#%%

class ClassificationLayer(torch.nn.Module):
    def __init__(self,dropout_rate=0):
        super(ClassificationLayer, self).__init__()
        self.Linear_1=torch.nn.Linear(512,256)
        self.Linear_2=torch.nn.Linear(256,1)
        self.dropout=torch.nn.Dropout(dropout_rate)
        
    def forward(self,input):
        hidden=self.Linear_1(input)
        hidden=self.dropout(hidden)
        
        output=torch.sigmoid(self.Linear_2(hidden)) #取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间，可以用来做二分类。
        return output
if __name__ == "__main__":
    image=ImageFeature.ExtractImageFeature()
    textfeature=TextFeature.ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    attribute=AttributeFeature.ExtractAttributeFeature()
    fuse=FuseAllFeature.ModalityFusion()
    final_classifier=ClassificationLayer()
    for text,text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        image_result,image_seq=image(image_feature)
        attribute_result,attribute_seq=attribute(attribute_index)
        text_result,text_seq=textfeature(text)

        output=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        result=final_classifier(output)
        predict=torch.round(result) #返回一个新张量，将输入input张量的每个元素舍入到最近的整数。


        print(result.shape) #torch.Size([32, 1])
        print(result) #tensor([[0.5030],[0.5031],[0.5025],[0.5036], [0.5027],[0.5035],[0.5020],...
        print(predict)#tensor([[1.],[1.],[1.],[1.],
        break
