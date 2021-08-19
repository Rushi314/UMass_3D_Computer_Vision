from trainMVShapeClassifier import trainMVShapeClassifier
from testMVImageClassifier import testMVImageClassifier
import pickle as p
import torch

train_path = './dataset/train'
test_path = './dataset/test'

# TRAIN
model, info = trainMVShapeClassifier(train_path, cuda=False, verbose=False)
#
# TO SAVE TIME for just testing code, uncomment the following 2 lines to load your pre-trained model
model = torch.load('model/model_epoch_19.pth', map_location=lambda storage, location: storage)["model"]

info = p.load(open("info.p", "rb"))

# TEST
testMVImageClassifier(test_path, model, info, pooling='mean', cuda=False, verbose=False)
testMVImageClassifier(test_path, model, info, pooling='max', cuda=False, verbose=False)