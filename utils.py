import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
def normalizeVoiceLen(y,normalizedLen):
    nframes=len(y)
    y = np.reshape(y,[nframes,1]).T

    if(nframes<normalizedLen):
        res=normalizedLen-nframes
        res_data=np.zeros([1,res],dtype=np.float32)
        y = np.reshape(y,[nframes,1]).T
        y=np.c_[y,res_data]
    else:
        y=y[:,0:normalizedLen]
    return y[0]

def getNearestLen(framelength,sr):
    framesize = framelength*sr

    nfftdict = {}
    lists = [32,64,128,256,512,1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])
    framesize = int(sortlist[0][0])
    return framesize

def get_mfcc(path,n_mfcc):
    path=path
    y,sr = librosa.load(path,sr=None)
    VOICE_LEN=32000

    N_FFT=getNearestLen(0.25,sr)

    y=normalizeVoiceLen(y,VOICE_LEN)

    mfcc_data=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=n_mfcc,n_fft=N_FFT,hop_length=int(N_FFT/4))
    return mfcc_data

def plot_confusion_matrix(y_true, y_pred, labels):
    plt.figure()
    num = len(labels)
    C = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(0,num))

    plt.matshow(C, cmap=plt.cm.Reds)


    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')



    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.xticks(range(num), labels=labels) # 将x轴或y轴坐标，刻度 替换为文字/字符
    plt.yticks(range(num), labels=labels)
    # plt.show()
def get_evaluation(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=None, average=None, sample_weight=None)
    recall = recall_score(y_true, y_pred, labels=None, average=None, sample_weight=None)
    f1 = f1_score(y_true, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
    precision = np.around(precision, 3)
    recall = np.around(recall, 3)
    f1 = np.around(f1, 3)
    return acc, precision, recall, f1