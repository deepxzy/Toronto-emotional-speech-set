import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from models import *
from Segdataset import SegDataset
import argparse
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'datasets', help='root of data')
parser.add_argument('--save_root', type=str, default=r'checkpoints', help='root of saved model.pth')
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--random_seed', type=int, default=1, help='random seed')
parser.add_argument('--model_kind', type=str, default='cnn', help='kind of model, i.e. cnn or rnn or transformer')
parser.add_argument('--is_shuffle_dataset', type=bool, default=True, help='shuffle dataset or not')
parser.add_argument('--test_split', type=float, default=0.25, help='ratio of the test set')
parser.add_argument('--n_mfcc', type=int, default=16, help='characteristic dimension of MFCC')
opt = parser.parse_args()

def main():
    if opt.model_kind == 'rnn':
        model = RNN(opt.n_mfcc)
    elif opt.model_kind == 'transformer':
        model = Transformer(opt.n_mfcc)
    elif opt.model_kind == 'cnn':
        model = CNN()

    if not os.path.exists(opt.save_root):
        os.makedirs(opt.save_root)
        print('create'+opt.save_root)

    train = SegDataset(root=opt.data_root, type='train', n_mfcc=opt.n_mfcc, random_state=opt.random_seed, test_size=opt.test_split)
    val = SegDataset(root=opt.data_root, type='val', n_mfcc=opt.n_mfcc, random_state=opt.random_seed, test_size=opt.test_split)

    train_iter = torch.utils.data.DataLoader(train, opt.batch_size, shuffle=opt.is_shuffle_dataset,
                                             drop_last=True, num_workers=opt.num_workers)
    test_iter = torch.utils.data.DataLoader(val, opt.batch_size, drop_last=True,
                                            num_workers=opt.num_workers)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    all_train_epoch_loss = []
    all_test_epoch_loss = []
    all_test_epoch_accuracy_emotion=[]
    all_test_epoch_accuracy_sex = []
    model = model.to(device)


    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), opt.lr)



    for epo in range(opt.epoch):
        train_loss = 0
        model.train()  # 启用batch normalization和drop out
        for index, (mfcc, emotion, sex) in enumerate(train_iter):
            mfcc, emotion, sex=mfcc.to(device), emotion.to(device), sex.to(device)
            optimizer.zero_grad()
            output_emotion,output_sex = model(mfcc)
            loss = criterion(output_emotion, emotion) + 0.1*criterion(output_sex, sex)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_iter), iter_loss))

        # test
        test_loss = 0
        correct_emotion, correct_sex = 0,0
        total=0
        model.eval()
        with torch.no_grad():
            for index, (mfcc, emotion, sex) in enumerate(test_iter):
                mfcc, emotion, sex = mfcc.to(device), emotion.to(device), sex.to(device)
                optimizer.zero_grad()
                output_emotion, output_sex = model(mfcc)
                loss = criterion(output_emotion, emotion) + 0.1*criterion(output_sex, sex)
                output_emotion = torch.argmax(output_emotion, dim=1)
                output_sex = torch.argmax(output_sex, dim=1)
                correct_emotion += (output_emotion == emotion).sum()
                correct_sex += (output_sex == sex).sum()
                total += len(emotion.view(-1))
                iter_loss = loss.item()
                test_loss += iter_loss
        accuracy_emotion = (correct_emotion / total).item()
        accuracy_sex = (correct_sex / total).item()
        print('<---------------------------------------------------->')
        print('epoch: %f' % epo)
        print('epoch train loss = %f, epoch test loss = %f,accuracy_emotion =%.3f,accuracy_sex =%.3f'
              % (train_loss / len(train_iter), test_loss / len(test_iter),accuracy_emotion,accuracy_sex))

        if np.mod(epo, 1) == 0:
            # 只存储模型参数
            torch.save(model.state_dict(), opt.save_root+'/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epo + 1), (train_loss / len(train_iter)), (test_loss / len(test_iter)))
                       )
            print('saving checkpoints/model_{}.pth'.format(epo))
        all_test_epoch_accuracy_emotion.append(accuracy_emotion)
        all_test_epoch_accuracy_sex.append(accuracy_sex)
        all_train_epoch_loss.append(train_loss / len(train_iter))
        all_test_epoch_loss.append(test_loss / len(test_iter))

    # plot
    plt.figure()
    plt.title('train_loss')
    plt.plot(all_train_epoch_loss)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root+'/train_loss.png')

    plt.figure()
    plt.title('test_loss')
    plt.plot(all_test_epoch_loss)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root+'/test_loss.png')

    plt.figure()
    plt.title('test_accury_emotion')
    plt.plot(all_test_epoch_accuracy_emotion)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root+'/test_accuracy_emotion.png')

    plt.figure()
    plt.title('test_accury_sex')
    plt.plot(all_test_epoch_accuracy_sex)
    plt.xlabel('epoch')
    plt.savefig(opt.save_root + '/test_accuracy_sex.png')
if __name__ == '__main__':
    main()
