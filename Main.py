from model_RAFP import *
from utilis import *

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import os
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

model_dir = "models\\"
directory = os.path.dirname(model_dir)
if not os.path.exists(directory):
    os.makedirs(directory)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=300)
checkpoint = ModelCheckpoint('models\\model-best.h5',
                            verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
## DATASET 1
train_dist = (300,300)
AFP_path = 'input/AFP_CKSAAP8.txt'
NON_AFP_path ='input/Non-AFP_CKSAAP8.txt'
X_train, Y_train, X_test, Y_test, test_dist, input_dim, nb_classes \
   = Generate_Datasets(AFP_path,NON_AFP_path,train_dist)


#model = RAFP_model(input_dim=input_dim, nb_classes=nb_classes)


model = RAFP_AC_modelu(input_dim=input_dim, nb_classes=nb_classes)  # THIS IS ONE MODEL, IT IS WORKING FINE I THINK ALL METRICS ARE OK #model = RAFP_model_Skip(input_dim=input_dim, nb_classes=nb_classes)

print("Training...")


history = model.fit({'enc_input': X_train},{'class_output': Y_train, 'decoder_output': X_train},epochs=1000, batch_size=1000, validation_split=0.1, verbose=2, callbacks= [checkpoint, es])


from keras.models import load_model
del model  # deletes the existing model
model = load_model('models\\model-best.h5')


print("Generating test predictions...")

print("FOR DATASET # 1")
[Y_pred,X_pred] = model.predict(X_test, verbose=0)
GenerateScore(Y_pred,Y_test)


print(history.history.keys())
plt.plot(history.history['class_output_acc'])
plt.plot(history.history['val_class_output_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.figure()
plt.plot(history.history['decoder_output_loss'])
plt.plot(history.history['val_decoder_output_loss'])
plt.title('model decoder loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

