import torch
from PIL import Image
import LoadData
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING_PATH = "C:/Users/86177/Desktop/pytorch-multimodal_sarcasm_detection-main/"
TEXT_LENGTH = 75
TEXT_HIDDEN = 256
token = BertTokenizer.from_pretrained('bert-base-uncased')
"""
read text file, find corresponding image path
"""

#用编号作为image的文件名、键，用"text": sentence, "group": group存储键值
def load_data():
    """
    dict() -> new empty dictionary
    """
    data_set = dict()
    """
    In training set, the last element in the list is the label.
    In validation and testing set, the last but one element in the list is labeled by hashtag(original post). 
    The last element in the list is labeled by another human annotator.
    The number of posts is bigger than that are used in the code. 
    Since in the code, posts with these words (exgag, sarcasm, sarcastic, reposting, , 
    joke, humour, humor, jokes, irony, ironic) are discarded. Images of these posts are not uploaded.
    """
    for dataset in ["train"]:
        file = open(os.path.join(WORKING_PATH, "text_data/", dataset + ".txt"), "rb")
        for line in file:
            content = eval(line)
            image = content[0]
            sentence = content[1]
            group = content[2]
            if os.path.isfile(os.path.join(WORKING_PATH, "image_data/", image + ".jpg")):
                data_set[int(image)] = {"text": sentence, "group": group}
    for dataset in ["test", "valid"]:
        file = open(os.path.join(WORKING_PATH, "text_data/", dataset + ".txt"), "rb")
        for line in file:
            content = eval(line)
            image = content[0]
            sentence = content[1]
            group = content[3]  # 2
            if os.path.isfile(os.path.join(WORKING_PATH, "image_data/", image + ".jpg")):
                data_set[int(image)] = {"text": sentence, "group": group}
    return data_set

#字典存储数据信息
data_set = load_data()

"""
load all training data 
"""


# load word index
def load_word_index():
    """
    在机器学习中，我们常常需要把训练好的模型存储起来，这样在进行决策时直接将模型读出，而不需要重新训练模型，这样就大大节约了时间。
    Python提供的pickle模块就很好地解决了这个问题，它可以序列化对象并保存到磁盘中，并在需要的时候读取出来，任何对象都可以执行序列化操作。
    """
    '''
    latin-1字符集是在ascii码上的一个扩展，它把ascii码没有用到过的字节码都给编上了对应的字符，所以它能表示

　　的字符就更多了；针对单个字节来说就没有它不能解码的，这个就是它的牛逼之处所在。也就是说当我们不在乎内容中多字节码的正确怕的情况

　　下使用latin-1字符集是不会出现解码异常的

    '''
    word2index = pickle.load(open(os.path.join(WORKING_PATH, "text_embedding/vocab.pickle"), 'rb'), encoding='latin1')
    return word2index

#{'raining': 6138, 'writings': 2902, '1,2': 2903, 'yellow': 2904, 'four': 6140, 'gag': 8216 ....}
word2index = load_word_index()

# load image labels
def load_image_labels():
    """返回每个图片的五个标签和所有标签索引"""
    # get labels
    img2labels = dict()
    with open(os.path.join(WORKING_PATH, "multilabel_database/", "img_to_five_words.txt"), "rb") as file:
        for line in file:
            content = eval(line)
            img2labels[int(content[0])] = content[1:]
    # label to index of embedding, dict, word:value 0~1001   pickle.load(file)  函数的功能：将file中的对象序列化读出。
    label2index = pickle.load(open(os.path.join(WORKING_PATH, "multilabel_database_embedding/vocab.pickle"), 'rb'))
    return img2labels, label2index

#第一个长度 36468   822592451823206402: ['red', 'car', 'parked', 'parking', 'lot'],.............
#第二个{'slope': 516, 'pointing': 3, 'hats': 4, 'people': 825, 'yellow': ....}
img2labels, label2index = load_image_labels()


# save to dataloader
class my_data_set(Dataset):
    def __init__(self, data):
        self.data = data
        self.image_ids = list(data.keys())
        for id in data.keys():         #数据集中增加一个键，键为image_path，存储文件地址
            self.data[id]["image_path"] = os.path.join(WORKING_PATH, "image_data/", str(id) + ".jpg")

        # load all text
        for id in data.keys():
            text = self.data[id]["text"].split()
            text_index = torch.empty(TEXT_LENGTH, dtype=torch.long)
            curr_length = len(text)
            """
            token_data = token.batch_encode_plus(batch_text_or_text_pairs=text,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=75,
                                           return_tensors='pt',
                                           return_length=True)
            self.data[id]["input_ids"] = token_data['input_ids']
            print(token_data['input_ids'].shape)
            self.data[id]["attention_mask"] = token_data['attention_mask']
            print(token_data['attention_mask'].shape)
            self.data[id]["token_type_ids"] = token_data['token_type_ids']
            print(token_data['token_type_ids'].shape
            """


            for i in range(TEXT_LENGTH):
                if i >= curr_length:
                    text_index[i] = word2index["<pad>"]  #text_index[i]句子中的每个单词
                elif text[i] in word2index:
                    text_index[i] = word2index[text[i]]
                else:
                    text_index[i] = word2index["<unk>"]
            self.data[id]["text_index"] = text_index   #数据集中增加一个键，键为text_index，存储每个句子中每个单词索引 [10016,   121,  9708,  7282,     0,     0,     0,     0,     0,...]



    # load image feature data - resnet 50 result  Size([32, 196, 2048]) 每张图片所有区域的特征
    def __image_feature_loader(self, id):
        attribute_feature = np.load(os.path.join(WORKING_PATH, "image_feature_data_temp", str(id) + ".npy"))
        return torch.from_numpy(attribute_feature)

    # load attribute feature data - 5 words label
    def __attribute_loader(self, id):
        """返回五个标签索引"""
        labels = img2labels[id]
        label_index = list(map(lambda label: label2index[label], labels))
        return torch.tensor(label_index)

    def __text_index_loader(self, id):
        return self.data[id]["text_index"]


    # # load text index
    # def __text_index_loader(self,id):
    #     text=self.data[id]["text"].split()
    #     text_index=torch.empty(TEXT_LENGTH,dtype=torch.long)
    #     curr_length=len(text)
    #     for i in range(TEXT_LENGTH):
    #         if i>=curr_length:
    #             text_index[i]=word2index["<pad>"]
    #         elif text[i] in word2index:
    #             text_index[i]=word2index[text[i]]
    #         else:
    #             text_index[i]=word2index["<unk>"]
    #     return text_index
    # load image

    def image_loader(self, id):
        path = self.data[id]["image_path"]
        img_pil = PIL.Image.open(path)
        transform = transforms.Compose([transforms.Resize((448, 448)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        img_tensor = transform(img_pil)
        return img_tensor

    def __text_loader(self, id):
        return self.data[id]["text"]

    def label_loader(self, id):
        return img2labels[id]


    def __getitem__(self, index):
        id = self.image_ids[index]
        # img = self.__image_loader(id)
        text = self.__text_loader(id)
        text_index = self.__text_index_loader(id)
        image_feature = self.__image_feature_loader(id)
        attribute_index = self.__attribute_loader(id)
        group = self.data[id]["group"]
        """
        input_ids = self.__input_ids_loader(id)
        attention_mask = self.__attention_mask_loader(id)
        token_type_ids = self.__token_type_ids_loader(id)
        """
        return text, text_index, image_feature, attribute_index, group, id

    def __len__(self):
        return len(self.image_ids)


def train_val_test_split(all_Data, train_fraction, val_fraction):
    # split the data
    train_val_test_count = [int(len(all_Data) * train_fraction), int(len(all_Data) * val_fraction), 0]
    train_val_test_count[2] = len(all_Data) - sum(train_val_test_count)
    return random_split(all_Data, train_val_test_count, generator=torch.Generator().manual_seed(42))



# train, val, test, split
all_Data = my_data_set(data_set)
train_fraction = 0.8
val_fraction = 0.1
batch_size = 32  # 32
train_set, val_set, test_set = train_val_test_split(all_Data, train_fraction, val_fraction)
# add to dataloader
# all_loader = DataLoader(all_Data,batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
play_loader = DataLoader(test_set, batch_size=1, shuffle=True)

"""
example of the dat\
a
"""
if __name__ == "__main__": #当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行
    #print(all_Data[1])
    #print(img2labels,"\n", label2index)
    #for i, (input_ids, attention_mask, token_type_ids) in enumerate(train_loader):
        #print(input_ids.shape, attention_mask.shape, token_type_ids.shape)
        #break


    for text, text_index, image_feature, attribute_index, group, id in train_loader:
        # plt.imshow(img[0].permute(1,2,0))
        # plt.show()
        #print(input_ids.shape, attention_mask.shape, token_type_ids.shape)
        print("text", text[0])
        print("text_index", text_index.shape, text_index.type())
        print("image feature", image_feature.shape, image_feature.type())
        print("attribute index", attribute_index)
        print("attribute index", attribute_index.shape, attribute_index.type())
        print("group", group, group.type())
        print("image id", id, id.type())
        break
"""
text_index torch.Size([32, 75]) torch.LongTensor
image feature torch.Size([32, 196, 2048]) torch.FloatTensor
attribute index tensor([[784, 759,  88, 935,   5],
        [ 63, 275, 679, 426, 640],
        [238, 384, 741, 548,  63],
        [238, 735,  97, 486, 140],
        [541, 788, 769, 340,  63],
        [238, 101,  63, 710, 541],
        [824, 825, 281, 610,  29],
        [503, 473,   5, 255, 551],
        [649, 968, 711, 523, 784],
        [101,  93,  85, 429, 825],
        [ 47, 579, 191, 769, 354],
        [281, 426, 279, 686, 541],
        [238, 722, 961, 778, 399],
        [246, 405, 968, 523, 769],
        [598, 670, 406, 203, 328],
        [161, 598, 968, 238, 465],
        [281, 825, 140, 195, 968],
        [713, 420, 930, 530,  89],
        [825, 814, 249, 410, 281],
        [ 63, 405,  20, 802, 968],
        [649, 759, 432, 975, 264],
        [140,  27, 341, 659, 902],
        [822, 961, 171, 140, 541],
        [ 18, 967, 826, 710, 548],
        [437, 684, 230, 807, 332],
        [805,  26, 505, 934, 679],
        [ 29,  27, 961, 238, 341],
        [ 71, 598, 968, 611, 779],
        [ 27, 140, 718, 238, 512],
        [997, 967,  27, 548, 797],
        [109, 140,  18, 355,  53],
        [611, 548, 793, 264, 238]])
attribute index torch.Size([32, 5]) torch.LongTensor
group tensor([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 1]) torch.LongTensor
image id tensor([898276305040539652, 709062525409693696, 823313136845520896,
        820417792008658945, 867793792367251456, 894320073032306695,
        822592842870784000, 818605360458174464, 822506680285921280,
        938217152422825984, 718191761370296320, 817518269775118336,
        822226795432968192, 821868109753647104, 821506052650831872,
        822230409677275136, 923836114745675776, 729032870250033152,
        822225772177801216, 910860301846831104, 730040802756415488,
        819324360334934016, 822955576703545344, 820413510760931328,
        822224374157561856, 822229874211483648, 820417707992432640,
        821862650451738624, 822954535605981186, 818607794203070464,
        820782269682089984, 826521720681013249]) torch.LongTensor
"""
