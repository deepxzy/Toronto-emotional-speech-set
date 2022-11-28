import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        self.fc1=nn.Linear(128,7)
        self.fc2=nn.Linear(128,2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x=x.unsqueeze(dim=1)
        x=self.cnn(x).squeeze()
        emotion=self.fc1(x)
        sex=self.fc2(x)
        emotion = self.softmax(emotion)
        sex = self.softmax(sex)

        return emotion,sex

class RNN(nn.Module):
    def __init__(self,n_mfcc=16):
        super().__init__()

        self.LSTM=nn.LSTM(n_mfcc, 40, 4,batch_first=True)
        self.avg=nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(40, 7)
        self.fc2 = nn.Linear(40, 2)
        self.softmax=nn.Softmax(dim=1)


    def forward(self, x):
        x=self.LSTM(x)[0].transpose(2,1)
        x=self.avg(x).squeeze()
        emotion = self.fc1(x)
        sex = self.fc2(x)
        emotion = self.softmax(emotion)
        sex = self.softmax(sex)
        return emotion, sex

class Transformer(nn.Module):
    def __init__(self,n_mfcc=16):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_mfcc, nhead=4, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(n_mfcc, 7)
        self.fc2 = nn.Linear(n_mfcc, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x=self.transformer_encoder(x).transpose(2,1)
        x = self.avg(x).squeeze()
        emotion = self.fc1(x)
        sex = self.fc2(x)
        emotion = self.softmax(emotion)
        sex = self.softmax(sex)
        return emotion, sex

class CNN_test(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        self.fc1=nn.Linear(128,7)
        self.fc2=nn.Linear(128,2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x=x.unsqueeze(dim=1)
        x=self.cnn(x).squeeze()
        x = x.unsqueeze(dim=0)
        emotion=self.fc1(x)
        sex=self.fc2(x)
        emotion = self.softmax(emotion).squeeze()
        sex = self.softmax(sex).squeeze()
        emotion = torch.argmax(emotion, dim=0)
        sex = torch.argmax(sex, dim=0)
        return emotion,sex

class RNN_test(nn.Module):
    def __init__(self,n_mfcc=16):
        super().__init__()

        self.LSTM=nn.LSTM(n_mfcc, 40, 4,batch_first=True)
        self.avg=nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(40, 7)
        self.fc2 = nn.Linear(40, 2)
        self.softmax=nn.Softmax(dim=1)


    def forward(self, x):
        x=self.LSTM(x)[0].transpose(2,1)
        x=self.avg(x).squeeze()
        x = x.unsqueeze(dim=0)
        emotion = self.fc1(x)
        sex = self.fc2(x)
        emotion = self.softmax(emotion).squeeze()
        sex = self.softmax(sex).squeeze()
        emotion = torch.argmax(emotion, dim=0)
        sex = torch.argmax(sex, dim=0)
        return emotion, sex

class Transformer_test(nn.Module):
    def __init__(self,n_mfcc=16):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_mfcc, nhead=4, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(n_mfcc, 7)
        self.fc2 = nn.Linear(n_mfcc, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x=self.transformer_encoder(x).transpose(2,1)
        x = self.avg(x).squeeze()
        x = x.unsqueeze(dim=0)
        emotion = self.fc1(x)
        sex = self.fc2(x)
        emotion = self.softmax(emotion).squeeze()
        sex = self.softmax(sex).squeeze()
        emotion = torch.argmax(emotion, dim=0)
        sex = torch.argmax(sex, dim=0)
        return emotion, sex
if __name__ == '__main__':
    input_tensor=torch.ones((1,126,16))
    model=Transformer_test()
    emotion,sex=model(input_tensor)
    print(emotion,sex)
