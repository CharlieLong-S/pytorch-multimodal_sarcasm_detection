#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


class Multimodel(torch.nn.Module):
    def __init__(self,fc_dropout_rate):
        super(Multimodel, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN)
        self.fuse = FuseAllFeature.ModalityFusion()
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
    def forward(self, text, image_feature, attribute_index):
        image_result,image_seq = self.image(image_feature)
        attribute_result,attribute_seq = self.attribute(attribute_index)
        text_result,text_seq = self.text(text)
        fusion = self.fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        output = self.final_classifier(fusion)
        return output
        


# In[3]:


def train(model,train_loader,val_loader,loss_fn,optimizer,number_of_epoch):
    for epoch in range(number_of_epoch):
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
        positive_number = 0
        pred_positive_number = 0
        pred_right_positive_number = 0
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
                flag = torch.full_like(val_pred, -1).to(torch.float32)
                a = (val_pred.round() == 1).to(torch.float32)
                b = torch.where(a == 1, a, flag).round()
                positive_number += (val_group == 1).sum().item()
                pred_right_positive_number += (b == val_group).sum().item()
                pred_positive_number += (val_pred.round() == 1).sum().item()

            P = pred_right_positive_number / pred_positive_number
            R = pred_right_positive_number / positive_number
            F = 2 * P * R / (P + R)

        print("epoch: %d train_loss=%.5f train_acc=%.3f valid_loss=%.5f valid_acc=%.3f Pre=%.4f Rec=%.4f F-score=%.4f" % (epoch,
                                                                                                              train_loss / len(train_loader),
                                                                                                              correct_train / len(train_loader) / batch_size,
                                                                                                              valid_loss / len(val_loader),
                                                                                                              correct_valid / len(val_loader) / batch_size,
                                                                                                              P, R, F))

        with open("tiaocan.txt", "a") as file:
            file.write("\n")
            file.write("epoch: %d train_loss=%.5f train_acc=%.3f valid_loss=%.5f valid_acc=%.3f"%(epoch,
                                                                                         train_loss/len(train_loader),
                                                                                      correct_train/len(train_loader)/batch_size,
                                                                                         valid_loss/len(val_loader),
                                                                                         correct_valid/len(val_loader)/batch_size))


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
    with open("tiaocan.txt", "a") as file:
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



# In[14]:


import sklearn.metrics as metrics
import seaborn as sns
def validation_metrics (model, dataset):
    model.eval()
    with torch.no_grad():
        correct=0
        confusion_matrix_sum=None
        loss_sum=0
        for text, text_index, image_feature, attribute_index, group, id in dataset:
            group = group.view(-1,1).to(torch.float32).to(device)
            pred = model(text, image_feature.to(device), attribute_index.to(device))
            loss = loss_fn(pred, group)
            loss_sum+=loss
            correct+=(pred.round()==group).sum().item()
            # calculate confusion matrix
            if confusion_matrix_sum is None:
                confusion_matrix_sum=metrics.confusion_matrix(group.to("cpu"),pred.round().to("cpu"),labels=[0,1])
            else:
                confusion_matrix_sum+=metrics.confusion_matrix(group.to("cpu"),pred.round().to("cpu"),labels=[0,1])
        acc=correct/len(dataset)/batch_size
        loss_avg=loss_sum/len(dataset)
    return loss_avg.item(), acc, confusion_matrix_sum

def plot_confusion_matrix(confusion_matrix):
    emotions=['not sarcasm','sarcasm']
    sns.heatmap(confusion_matrix, annot=True, xticklabels=emotions, yticklabels=emotions, fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
loss, acc, confusion_matrix=validation_metrics (model, test_loader)
print("loss:",loss,"accuracy:",acc)
plot_confusion_matrix(confusion_matrix)


# In[21]:


import matplotlib.pyplot as plt
def validation_metrics (model, dataset):
    model.eval()
    with torch.no_grad():
        count=0
        for text, text_index, image_feature, attribute_index, group, id in dataset:
            if count==5:
                break
            id=id.item()
            print(f">>>Example {count+1}<<<")
            img=all_Data.image_loader(id)
            plt.imshow(img.permute(1,2,0))
            plt.show()
            print("Text: ",all_Data.__text_loader(id))
            print("Labels: ",all_Data.label_loader(id))
            print(f"Truth:{' not ' if group[0]==0 else ' '}sarcasm")
            pred = model(text, image_feature.to(device), attribute_index.to(device))
            print(f"Preduct:{' not ' if round(pred[0,0].item())==0 else ' '}sarcasm")
            count+=1

validation_metrics (model, play_loader)




# In[ ]:




