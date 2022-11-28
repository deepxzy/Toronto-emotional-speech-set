import matplotlib.pyplot as plt
from models import *
from Segdataset import SegDataset
import argparse
from utils import plot_confusion_matrix, get_evaluation
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'datasets', help='root of data')
parser.add_argument('--log_root', type=str, default=r'checkpoints/Transformer/Transformer.pth', help='root of model.pth')
parser.add_argument('--save_root', type=str, default=r'checkpoints/Transformer', help='root of saved confusion_matrix')
parser.add_argument('--random_seed', type=int, default=10, help='random seed')
parser.add_argument('--model_kind', type=str, default='transformer', help='kind of model, i.e. cnn or rnn or transformer')
parser.add_argument('--test_split', type=float, default=0.25, help='ratio of the test set')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--n_mfcc', type=int, default=16, help='characteristic dimension of MFCC')
parser.add_argument('--is_plt', type=bool, default=False, help='plt or not')
opt = parser.parse_args()

def main():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']
    sexs=['OAF', 'YAF']
    if opt.model_kind == 'rnn':
        model = RNN_test(opt.n_mfcc)
    elif opt.model_kind == 'transformer':
        model = Transformer_test(opt.n_mfcc)
    elif opt.model_kind == 'cnn':
        model = CNN_test()
    output_emotion_list=[]
    output_sex_list=[]
    label_emotion_list=[]
    label_sex_list=[]
    if not os.path.exists(opt.save_root):
        os.makedirs(opt.save_root)
        print('create'+opt.save_root)


    val = SegDataset(root=opt.data_root, type='val', n_mfcc=opt.n_mfcc, random_state=opt.random_seed, test_size=opt.test_split)


    test_iter = torch.utils.data.DataLoader(val, batch_size=1, drop_last=True,
                                            num_workers=opt.num_workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.load_state_dict(torch.load(opt.log_root,map_location=device))

    model.eval()
    with torch.no_grad():
        for index, (mfcc, emotion, sex) in enumerate(test_iter):
            mfcc, emotion, sex = mfcc.to(device), emotion.to(device), sex.to(device)

            output_emotion, output_sex = model(mfcc)
            output_emotion, output_sex = output_emotion.cpu().numpy(), output_sex.cpu().numpy()
            emotion, sex = emotion.cpu().numpy(), sex.cpu().numpy()

            output_emotion_list.append(output_emotion)
            output_sex_list.append(output_sex)
            label_emotion_list.append(emotion)
            label_sex_list.append(sex)

    acc, precision, recall, f1 = get_evaluation(label_emotion_list, output_emotion_list)
    with open(os.path.join(opt.save_root, 'evaluation.txt'), 'w') as f:
        f.write('accuracy: ' + str(acc))
        f.write('\r\n')
        f.write('precision: ' + str(precision))
        f.write('\r\n')
        f.write('recall: ' + str(recall))
        f.write('\r\n')
        f.write('f1 score: ' + str(f1))


    plot_confusion_matrix(label_emotion_list, output_emotion_list, emotions)
    plt.savefig(opt.save_root + '/confusion_matrix_emotion.png')
    plot_confusion_matrix(label_sex_list, output_sex_list, sexs)
    plt.savefig(opt.save_root + '/confusion_matrix_sex.png')


if __name__ == '__main__':
    main()
