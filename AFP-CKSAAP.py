# CKSAAP_AFP: <title of your paper>
# CItation to your paper (I will tell you how later)
# some decription of required packages e.g. what are the requirements ok your CKSAAP
from keras.models import load_model
from utilis import *
import h5py

hf = h5py.File('data.h5', 'r')
X_train1 = hf.get('dataset_1')
X_test1 = hf.get('dataset_2')
Y_train1 = hf.get('dataset_3')
Y_test1 = hf.get('dataset_4')


model = load_model('my_model.h5')

print("Generating test predictions...")

print("FOR DATASET # 1")
Y_pred1 = model.predict(X_test1, verbose=0)
GenerateScore(Y_pred1,Y_test1)
