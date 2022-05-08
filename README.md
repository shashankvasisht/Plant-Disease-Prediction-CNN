# Plant-Disease-Prediction-CNN
A CNN model to classify different plant diseases. Vgg16 net is fine tuned to the crowdAI dataset.
The dataset can be found at https://www.crowdai.org/challenges/1

# Requirements
- python 2.7
- numpy
- tensorflow
- keras
- OpenCV (can use pillow Library Too)


# Methodology
The pre-trained VGG16 model trained on imagenet dataset was used and the last 4 convolutional layers were trained on our dataset. This process is called fine tuning.
A point to note is that only 19 classes were used from the dataset since we were focussing on a particular region where only these crops are grown.
