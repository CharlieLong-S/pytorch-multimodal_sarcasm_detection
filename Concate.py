from torch.nn import functional as F
import torch.nn as nn
import torchvision
import torch
import ImageFeature
import AttributeFeature
import TextFeature
import FinalClassifier
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Multimodel(nn.Module):

    def __init__(self, dropout_rate=0):
        super(Multimodel, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN)

        # create a new classifier
        """self.model1.fc = nn.Identity()
        self.model2.fc = nn.Identity()
        self.model3.fc = nn.Identity()"""
        self.Linear_1 = torch.nn.Linear(1212, 512)
        self.Linear_2 = torch.nn.Linear(512, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)


    def forward(self, text, image_feature, attribute_index, mode=None):
        image_result, image_seq = self.image(image_feature)
        attribute_result, attribute_seq = self.attribute(attribute_index)
        text_result, text_seq = self.text(text)

        input = torch.cat((image_result, attribute_result, text_result), dim=1)
        hidden = self.Linear_1(input)
        hidden = self.dropout(hidden)

        output = torch.sigmoid(self.Linear_2(hidden))  # 取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间，可以用来做二分类。
        return output

def train(model,train_loader,val_loader,loss_fn,optimizer,number_of_epoch):
    for epoch in range(number_of_epoch): #iteration: 1个iteration 等于使用batchsize个样本训练一次   epoch: 1个epoch等于使用训练集中的全部样本训练一次
        #print("epoch")
        train_loss=0
        correct_train=0
        model.train() #model.train()的作用是启用Batch Normalization 和Dropout。 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。 model.train()是保证BN层能够用到每一批数据的均值和方差。
        for text, text_index, image_feature, attribute_index, group, id in train_loader:
            #print(id)
            group = group.view(-1,1).to(torch.float32).to(device) #真实值
            pred = model(text, image_feature.to(device), attribute_index.to(device))
            loss = loss_fn(pred, group)
            train_loss+=loss
            correct_train+=(pred.round()==group).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate valid loss

        valid_loss=0
        correct_valid=0
        positive_number=0
        pred_positive_number=0
        pred_right_positive_number=0

        model.eval()
        with torch.no_grad():
            for val_text,val_text_index, val_image_feature, val_attribute_index, val_group, val_id in val_loader:

                val_group = val_group.view(-1,1).to(torch.float32).to(device)
                val_pred = model(val_text, val_image_feature.to(device), val_attribute_index.to(device))
                val_loss = loss_fn(val_pred, val_group)
                valid_loss+=val_loss
                correct_valid+=(val_pred.round()==val_group).sum().item()
                # Pre 精度是精确性的度量，表示被分为正例的示例中，实际为正例的比例：
                # Rec 灵敏度表示所有正例中，被分对的比例， 灵敏度与召回率Recall计算公式完全一致
                flag=torch.full_like(val_pred, -1).to(torch.float32)
                a=(val_pred.round() == 1).to(torch.float32)
                b=torch.where(a == 1, a, flag).round()
                positive_number += (val_group == 1).sum().item()
                pred_right_positive_number += (b==val_group).sum().item()
                pred_positive_number += (val_pred.round() == 1).sum().item()


        P=pred_right_positive_number/pred_positive_number
        R=pred_right_positive_number/positive_number
        F=2*P*R/(P+R)

        print("epoch: %d train_loss=%.5f train_acc=%.3f valid_loss=%.5f valid_acc=%.3f Pre=%.4f Rec=%.4f F-score=%.4f"%(epoch,
                                                                                         train_loss/len(train_loader),
                                                                                      correct_train/len(train_loader)/batch_size,
                                                                                         valid_loss/len(val_loader),
                                                                                         correct_valid/len(val_loader)/batch_size,
                                                                                          P,R,F))
        with open("tiaocan_concate.txt", "a") as file:
            file.write("\n")
            file.write("epoch: %d train_loss=%.5f train_acc=%.3f valid_loss=%.5f valid_acc=%.3f Pre=%.4f Rec=%.4f F-score=%.4f"%(epoch,
                                                                                         train_loss/len(train_loader),
                                                                                      correct_train/len(train_loader)/batch_size,
                                                                                         valid_loss/len(val_loader),
                                                                                         correct_valid/len(val_loader)/batch_size,
                                                                                          P,R,F))


# In[22]:


learning_rate_list = [0.001]
fc_dropout_rate_list=[0,0.3,0.9,0.99]
#lstm_dropout_rate_list=[0, 0.2, 0.4]
weight_decay_list=[0,1e-6,1e-5,1e-4]
# weight_decay_list=[1e-7]
batch_size=32
data_shuffle=False


# In[23]:


# load data
train_fraction=0.8
val_fraction=0.1
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=data_shuffle)
val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=data_shuffle)
test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=data_shuffle)
play_loader = DataLoader(test_set,batch_size=1, shuffle=data_shuffle)


# In[ ]:


# start train
import itertools
comb = itertools.product(learning_rate_list, fc_dropout_rate_list,weight_decay_list)
for learning_rate, fc_dropout_rate,weight_decay in list(comb):
    print(f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} |  weight decay={weight_decay}")
    with open("tiaocan_concate.txt", "a") as file:
        file.write("\n")
        file.write(f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} |  weight decay={weight_decay}")
    # loss function
    loss_fn=torch.nn.BCELoss()
    # initilize the model
    model = Multimodel(fc_dropout_rate).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    # train
    number_of_epoch=7
    train(model,train_loader,val_loader,loss_fn,optimizer,number_of_epoch)





