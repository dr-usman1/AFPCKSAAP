import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization

def Generate_Datasets(AFP_path,NON_AFP_path,train_dist):
    # Read data
    AFP = pd.read_csv(AFP_path,sep='\t') # our csv file use 'tab:\t' as delimeter
    # Removing protein ids and extracting attributes
    AFP = (AFP.ix[:,1:].values).astype('float32')

    NON_AFP = pd.read_csv(NON_AFP_path,sep='\t')# our csv file use 'tab:\t' as delimeter
    # Removing protein ids and extracting attributes
    NON_AFP = (NON_AFP.ix[:,1:].values).astype('float32')

    AFP_pool = np.random.permutation(AFP.shape[0])
    NON_AFP_pool = np.random.permutation(NON_AFP.shape[0])

    num_features = NON_AFP.shape[1]

    if ((train_dist[0]>0) and (train_dist[1]>0)):
       X_train = np.zeros((np.sum(train_dist), num_features))
       X_train[0:train_dist[0]] = AFP[AFP_pool[0:train_dist[0]]]
       X_train[train_dist[0]:train_dist[0] + train_dist[1]] = NON_AFP[NON_AFP_pool[0:train_dist[1]]]
       tr_labels = np.concatenate((np.ones(train_dist[0]), np.zeros(train_dist[1])), axis=0)
       Y_train = np_utils.to_categorical(tr_labels)

    test_dist =  (AFP.shape[0]-train_dist[0],NON_AFP.shape[0]-train_dist[1])

    X_test = np.zeros((np.sum(test_dist), num_features))

    X_test[0:test_dist[0]] = AFP[AFP_pool[train_dist[0]:train_dist[0]+test_dist[0]]]
    X_test[test_dist[0]:test_dist[0]+test_dist[1]] = NON_AFP[NON_AFP_pool[train_dist[1]:train_dist[1]+test_dist[1]]]

    ts_labels = np.concatenate((np.ones(test_dist[0]),np.zeros(test_dist[1])),axis=0)
    Y_test = np_utils.to_categorical(ts_labels)

    if ((train_dist[0] > 0) and (train_dist[1] > 0)):
        print(" VERIFICATION")
        print(" num of features :",num_features,"\n",
              "number of AFPs :", AFP.shape[0],"\n",
              "number of NON_AFPs :", NON_AFP.shape[0],"\n",
              "train data (AFPs, NON_AFPs) :", train_dist,"\n",
              "number of AFPs :", test_dist, "\n",
              "train data size (AFPs+NON_AFPs, number of features) : ",
              np.sum(train_dist),num_features,"=", X_train.shape, "\n",
              "number of AFPs :", test_dist, "\n",
              "test data size (AFPs+NON_AFPs, number of features) : ",
              np.sum(test_dist), num_features, "=", X_test.shape, "\n",
              "train label size (AFPs+NON_AFPs, number of classes) : ",
              np.sum(train_dist),num_features,"=", Y_train.shape, "\n",
              "number of AFPs :", test_dist, "\n",
              "test data size (AFPs+NON_AFPs, number of classes) : ",
              np.sum(test_dist), num_features, "=", Y_test.shape)
        X_train = np.expand_dims(X_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    nb_classes = Y_test.shape[1]

    if ((train_dist[0] > 0) and (train_dist[1] > 0)):
        return X_train, Y_train, X_test, Y_test, test_dist, num_features, nb_classes;
    else:
        return X_test, Y_test, test_dist, num_features, nb_classes;

def GenerateScore(Y_pred,Y_target):
    class_indices = np.argmax(Y_target, axis=1)
    # y_preds = np_utils.to_categorical(preds)
    predicted_class_indices = np.argmax(Y_pred, axis=1)

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    cm = confusion_matrix(class_indices, predicted_class_indices)
    TN = np.double(cm[0][0])
    FN = np.double(cm[1][0])
    TP = np.double(cm[1][1])
    FP = np.double(cm[0][1])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # print('Sensitivity is ', str(sensitivity))
    # print('Specificity is ', str(specificity))
    # print('Accuracy is ', str(accuracy))
    # print('Youden Index is ', str(sensitivity + specificity - 1))
    # print('MCC is ', str(MCC))
    # print("CLOSE ROC-Curve Figure")

    print(str(sensitivity))
    print(str(specificity))
    print(str(accuracy))
    print(str(sensitivity + specificity - 1))
    print(str(MCC))


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_target.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()

