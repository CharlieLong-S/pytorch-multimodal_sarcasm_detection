
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_PATH="C:/Users/86177/Desktop/pytorch-multimodal_sarcasm_detection-main/"
TEXT_LENGTH=75
TEXT_HIDDEN=256
"""
read text file, find corresponding image path
"""
def load_data():
    """is simillar with the load_data of LoadDate.py"""
    data_set=dict()
    for dataset in ["train"]:
        file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[2]
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set[int(image)]={"text":sentence,"group":group}
    for dataset in ["test","valid"]:
        file=open(os.path.join(WORKING_PATH,"text_data/",dataset+".txt"),"rb")
        for line in file:
            content=eval(line)
            image=content[0]
            sentence=content[1]
            group=content[3] #2
            if os.path.isfile(os.path.join(WORKING_PATH,"image_data/",image+".jpg")):
                data_set[int(image)]={"text":sentence,"group":group}
    return data_set

data_set=load_data()
"""
load image data
"""
image_feature_folder="image_feature_data_temp"
# pretrain dataloader
class pretrain_data_set(Dataset):
    def __init__(self, data):
        self.data=data
        self.image_ids=list(data.keys()) #图片目录
        for id in data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"image_data/",str(id)+".jpg")
    # load image
    def __image_loader(self,id):
        """将指定id图片的大小resize为(448,448)，并返回张量"""
        path=self.data[id]["image_path"]
        img_pil =  PIL.Image.open(path) #PIL打开图片通道顺序为RGB
        #torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起：
        transform = transforms.Compose([transforms.Resize((448,448)),  #PILImage对象size属性返回的是w, h，而resize的参数顺序是h, w。
                                        transforms.ToTensor(),#transform.ToTensor()  1. 是将输入的数据shape W，H，C ——> C，H，W  2. 将所有数除以255，将数据归一化到【0，1】
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        img_tensor = transform(img_pil)
        return img_tensor
    def __getitem__(self, index):
        """根据列表中的索引，返回图片id、和张量"""
        id=self.image_ids[index]
        image=self.__image_loader(id)
        return id,image
    def __len__(self):
        return len(self.image_ids)

sub_image_size=32 #448/14  使用一个预训练和微调的ResNet模型来获得图片的14*14区域向量
sub_graph_preprocess = transforms.Compose([
    transforms.ToPILImage(mode=None),
    transforms.Resize(256),
    transforms.ToTensor(),  #transform.ToTensor()  1. 是将输入的数据shape W，H，C ——> C，H，W  2. 将所有数除以255，将数据归一化到【0，1】
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #这个函数的输出output[channel] = (input[channel] - mean[channel]) / std[channel]。
    # 这里[channel]的意思是指对特征图的每个通道都进行这样的操作。第一个参数（0.5，0.5，0.5）表示每个通道的均值都是0.5，第二个参数（0.5，0.5，0.5）表示每个通道的方差都为0.5。
    #很多代码里面是这样的：torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])这一组值是怎么来的？这一组值是从imagenet训练集中抽样算出来的。
    # 经过上面normalize()的变换后变成了均值为0 方差为1（其实就是最大最小值为1和-1）每个样本图像变成了均值为0  方差为1 的标准正态分布，这就是最普通（科学研究价值最大的）的样本数据了
])
all_pretrain_dataset=pretrain_data_set(data_set)
"""
generate data
"""
class Identity(torch.nn.Module):
    """残差神经网络中的identity mapping 恒等映射 """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def densenet121_predictor():
    # extract the input for last fc layer（全连接层） in resenet50
    densenet121=torchvision.models.densenet121(pretrained=True) #返回在ImageNet上训练好的模型。
    for param in densenet121.parameters():
        param.requires_grad = False
    densenet121.fc = Identity()   #全连接层
    densenet121 = densenet121.to(device)  #使用model=model.to(device)，将模型加载到相应的设备中
    densenet121.eval()       #model.eval()的作用是不启用 Batch Normalization 和 Dropout。
    # save the output in .npy file
    densenet121_output_path=os.path.join(WORKING_PATH,image_feature_folder)
    if not os.path.exists(densenet121_output_path):
        os.makedirs(densenet121_output_path)

    with torch.no_grad():  #将原始向量平均，with torch. no_grad() 是一个上下文管理器，由它管理的代码块不需要计算梯度，也不会进行反向传播，
        # 因此在训练阶段常用于验证集计算loss、在测试阶段，则需要加上该代码，避免进行损失梯度的计算.
        total=len(all_pretrain_loader)*all_pretrain_loader.batch_size
        count=0
        time_s=time.perf_counter()  #计时器
        #第一个存储image的编号 torch.Size([64]) torch.Size([64, 3, 448, 448])
        for img_index,img in all_pretrain_loader:
            # seperate img(448,448) into 14*14 images with size (32,32)
            # [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            # [14,15,16,17,18,................]
            # [28,...]
            # ...
            # [182,....,195]
            sub_img_output=list() #每个图片小块区域
            for column in range(14):
                for row in range(14):
                    # resize image from (32,32) to (256,256)
                    #Size([64, 3, 32, 32])
                    sub_image_original=img[:,:,sub_image_size*row:sub_image_size*(row+1),sub_image_size*column:sub_image_size*(column+1)]
                    #Size([64, 3, 256, 256]) 每张图片小块区域
                    sub_image_normalized=torch.stack(list(map(lambda image:sub_graph_preprocess(image),sub_image_original)),dim=0)
                    output=densenet121(sub_image_normalized.to(device))
                    #(64, 1000)
                    sub_img_output.append(output.to("cpu").numpy())
            # (196, 64, 1000) to (64, 196, 1000)
            sub_img_output=np.array(sub_img_output).transpose([1,0,2])
            # save averaged attribute to "densenet121_output", same name as the image
            #0 tensor(840006160660983809)    1 tensor(908913372199915520)
            #将图片每块(32,32)特征存储在一个文件中
            for index,sub_img_index in enumerate(img_index):
                np.save(os.path.join(densenet121_output_path,str(sub_img_index.item())),sub_img_output[index])
            time_e=time.perf_counter()
            count+=all_pretrain_loader.batch_size
            total_time=time_e-time_s
            print(f"Completed {count}/{total} time left={int((total-count)*total_time/count/60/60)}:{int((total-count)*total_time/count/60%60)}:{int((total-count)*total_time/count%60)} speed={round(total_time/count,3)}sec/image")


# 32 is the minimum batch size can achieve best performance
all_pretrain_loader = DataLoader(all_pretrain_dataset,batch_size=64)

# it will take really long time to run...
#densenet121_predictor()
"""
test the image split
"""
"""
if __name__ == "__main__":
    # can be used to create image feature data
    # densenet121_predictor()
     for img_index,img in all_pretrain_loader:
         temp_img=img
         print(img[0].size())
         plt.imshow(img[0].permute(1,2,0))
         plt.show()
         print("======================================")
         # try to seperate
         for column in range(14):
             for row in range(14):
                 sub_index=row*14+column
                 sub_image_original=img[0][:,sub_image_size*row:sub_image_size*(row+1),sub_image_size*column:sub_image_size*(column+1)]
                 sub_image_normalized=sub_graph_preprocess(sub_image_original)
                 # show original sub image
                 plt.imshow(sub_image_original.permute(1,2,0))
                 plt.show()
                 # show normalized sub image
                 plt.imshow(sub_image_normalized.permute(1,2,0))
                 plt.show()
                 print(sub_index)
                 print(sub_image_original.size())
                 print(sub_image_normalized.size())
                 break
             break
         break
"""

if __name__ == "__main__":
    densenet121 = torchvision.models.densenet121(pretrained=True)  # 返回在ImageNet上训练好的模型。
    for param in densenet121.parameters():
        param.requires_grad = False
    densenet121.fc = Identity()  # 全连接层
    densenet121 = densenet121.to(device)  # 使用model=model.to(device)，将模型加载到相应的设备中
    densenet121.eval()  # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
    # save the output in .npy file
    densenet121_output_path = os.path.join(WORKING_PATH, image_feature_folder)
    if not os.path.exists(densenet121_output_path):
        os.makedirs(densenet121_output_path)

    with torch.no_grad():  # 将原始向量平均，with torch. no_grad() 是一个上下文管理器，由它管理的代码块不需要计算梯度，也不会进行反向传播，
        # 因此在训练阶段常用于验证集计算loss、在测试阶段，则需要加上该代码，避免进行损失梯度的计算.
        total = len(all_pretrain_loader) * all_pretrain_loader.batch_size
        count = 0
        time_s = time.perf_counter()  # 计时器
        # 第一个存储image的编号 torch.Size([64]) torch.Size([64, 3, 448, 448])
        for img_index, img in all_pretrain_loader:
            print("img_index_size:",img_index.shape,"\n")
            print(img_index[0],"\n\n")

            print("img_size:", img.shape, "\n")
            print(img[0], "\n\n")


            sub_img_output = list()
            for column in range(14):
                for row in range(14):
                    # resize image from (32,32) to (256,256)
                    # Size([64, 3, 32, 32])
                    sub_image_original = img[:, :, sub_image_size * row:sub_image_size * (row + 1),
                                         sub_image_size * column:sub_image_size * (column + 1)]

                    # Size([64, 3, 256, 256])
                    sub_image_normalized = torch.stack(
                        list(map(lambda image: sub_graph_preprocess(image), sub_image_original)), dim=0)

                    output = densenet121(sub_image_normalized.to(device))
                    # (64, 2048)
                    sub_img_output.append(output.to("cpu").numpy())
            # (196, 64, 2048) to (64, 196, 2048)
            sub_img_output = np.array(sub_img_output).transpose([1, 0, 2])
            print(sub_img_output.shape)
            # save averaged attribute to "densenet121_output", same name as the image
            # 0 tensor(840006160660983809)    1 tensor(908913372199915520)
            # 将每(32,32)图片的特征存储在
            for index, sub_img_index in enumerate(img_index):
                print(index,":",sub_img_index)
                #np.save(os.path.join(densenet121_output_path, str(sub_img_index.item())), sub_img_output[index])
            time_e = time.perf_counter()
            count += all_pretrain_loader.batch_size
            total_time = time_e - time_s
            break


