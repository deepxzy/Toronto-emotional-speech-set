import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import get_mfcc
emotions=['angry','disgust','fear','happy','neutral','ps','sad']
def read_file_list(root=r'datasets', type='train', n_mfcc=16, random_state=1, test_size=0.25):
    root = os.path.join(root, 'train')
    file_path_all=[]
    for dir in os.listdir(root):
        dir_root = os.path.join(root, dir)
        for file in os.listdir(dir_root):
            file_path = os.path.join(dir_root, file)
            file_path_all.append(file_path)
    mfcc_list=[]
    emotion_list=[]
    sex_list=[]
    for i in range(len(file_path_all)):
        mfcc=get_mfcc(file_path_all[i],n_mfcc)
        if 'OAF' in file_path_all[i]:
            sex=0
        elif 'YAF' in file_path_all[i]:
            sex=1
        for j in range(len(emotions)):
            if emotions[j] in file_path_all[i]:
                emotion=j
        mfcc_list.append(mfcc)
        emotion_list.append(emotion)
        sex_list.append(sex)
    mfcc_train, mfcc_val, emotion_train, emotion_val, sex_train, sex_val = train_test_split(mfcc_list, emotion_list, sex_list, test_size=test_size,
                                                                            random_state=random_state)
    # images_train,labels_train=images, labels
    if type=='train':
        return mfcc_train,  emotion_train, sex_train  # 两者路径的列表
    if type=='val':
        return mfcc_val, emotion_val, sex_val




class SegDataset(torch.utils.data.Dataset):
    def __init__(self, root=r'datasets', type='train', n_mfcc=16, random_state=1, test_size=0.25):

        mfcc, emotion, sex = read_file_list(root=root, type=type, n_mfcc=n_mfcc, random_state=random_state, test_size=test_size)

        self.mfcc = mfcc
        self.emotion = emotion
        self.sex=sex

        print('Read ' + str(len(self.mfcc)) + ' valid examples')


    def __getitem__(self, idx):
        mfcc = self.mfcc[idx]
        emotion = self.emotion[idx]
        sex = self.sex[idx]

        mfcc = torch.from_numpy(np.array(mfcc)).type(torch.FloatTensor).transpose(1,0)
        emotion = torch.from_numpy(np.array(emotion)).long()
        sex = torch.from_numpy(np.array(sex)).long()


        return mfcc, emotion, sex  # float32 tensor, uint8 tensor
#############################################
    def __len__(self):
        return len(self.mfcc)

#label === np.argmax(dlabel, axis=0)，二者可以互换
if __name__ == "__main__":

    voc_train = SegDataset()
    print(type(voc_train))#<class '__main__.VOCSegDataset'>
    print(len(voc_train))
    img, label,_ = voc_train[11]
    # img=np.transpose(np.array(img, np.float64), [1, 2, 0])
    print(img)
    print(label)
    # plt.imshow(img)
    # plt.show()
    print(type(img), type(label))
    print(img.shape, label.shape, _.shape)
    print(label,_)

