# Toronto-emotional-speech-set
Speech emotion and sex detection by deep learning (cnn, rnn, transformer) based on Toronto emotional speech set

## Model
We use first use MFCC to get the features, then use CNN, RNN, Transformer to train the datasets

## Datasets
The datasets can be download at 'https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess'

We do not use the test set. We only train and test by dividing the training set

## Train
run train.py 

## Test
run test.py

## Result
### CNN
![image](https://github.com/deepxzy/Toronto-emotional-speech-set/blob/main/checkpoints/CNN/confusion_matrix_emotion.png)
![image](https://github.com/deepxzy/Toronto-emotional-speech-set/blob/main/checkpoints/CNN/confusion_matrix_sex.png)
### RNN
![image](https://github.com/deepxzy/Toronto-emotional-speech-set/blob/main/checkpoints/RNN/confusion_matrix_emotion.png)
![image](https://github.com/deepxzy/Toronto-emotional-speech-set/blob/main/checkpoints/RNN/confusion_matrix_sex.png)
### Transformer
![image](https://github.com/deepxzy/Toronto-emotional-speech-set/blob/main/checkpoints/Transformer/confusion_matrix_emotion.png)
![image](https://github.com/deepxzy/Toronto-emotional-speech-set/blob/main/checkpoints/Transformer/confusion_matrix_sex.png)
