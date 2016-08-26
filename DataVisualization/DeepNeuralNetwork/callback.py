#!/usr/bin/env python2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cPickle
import numpy as np

import operator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import sys
import denali

def dnn_model_train(nepoch, nfeatures, X, y):
    model = Sequential()
    model.add(Dense(output_dim=100, input_dim=nfeatures, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(X, y, nb_epoch=nepoch, batch_size=250, verbose=0)
    return model

def compute_error_rate(y_answer, sentiment_list):
    y_test = []
    for sentiment in sentiment_list:
        index, value = max(enumerate(sentiment), key=operator.itemgetter(1))
        y_test.append(index)
    wrong = 0
    err_index_lst = []
    for i in xrange(len(y_answer)):
        #print y_train[i],y_test[i]
        if y_answer[i] != y_test[i]:
            wrong += 1
            err_index_lst.append(i)
    print float(wrong)/len(y_answer)
    return err_index_lst
    
def main():
    X_pos = cPickle.load(open('raw_data/pos_vects.pkl','rb'))
    X_neg = cPickle.load(open('raw_data/neg_vects.pkl','rb'))
    X_neu = cPickle.load(open('raw_data/neu_vects.pkl','rb'))

    X_pos_train = X_pos[0:275]
    X_neg_train = X_neg[0:500]
    X_neu_train = X_neu[0:250]

    X_train = X_pos_train + X_neg_train + X_neu_train

    y_pos_train = [1]*len(X_pos_train) + [0]*len(X_neg_train) + [0]*len(X_neu_train)
    y_neg_train = [0]*len(X_pos_train) + [1]*len(X_neg_train) + [0]*len(X_neu_train)
    y_neu_train = [0]*len(X_pos_train) + [0]*len(X_neg_train ) + [1]*len(X_neu_train)


    X_pos_test = X_pos[275:325]
    X_neg_test = X_neg[500:664]
    X_neu_test = X_neu[250:299]

    X_test = X_pos_test + X_neg_test + X_neu_test

    y_pos_test = [1]*len(X_pos_test) + [0]*len(X_neg_test) + [0]*len(X_neu_test)
    y_neg_test = [0]*len(X_pos_test) + [1]*len(X_neg_test) + [0]*len(X_neu_test)
    y_neu_test = [0]*len(X_pos_test) + [0]*len(X_neg_test ) + [1]*len(X_neu_test)

    y_pos_train = np.array(y_pos_train)
    y_neg_train = np.array(y_neg_train)
    y_neu_train = np.array(y_neu_train)
    y_pos_test = np.array(y_pos_test)
    y_neg_test = np.array(y_neg_test)
    y_neu_test = np.array(y_neu_test)
    n_feature = len(X_train[0])

    pca = PCA(n_components=3)
    X_train_pca = pca.fit(X_train).transform(X_train)
    print len(X_train_pca)
    X_test_pca = pca.fit(X_test).transform(X_test)
    print len(X_test_pca)

    selection = denali.io.read_selection_file(sys.argv[1])

    # get the id of the child of the component
    child = int(selection['component'][1][0])
    print child

    nb_epoch = child + 1
    model_pos = dnn_model_train(nepoch=nb_epoch, nfeatures=n_feature, X=X_train, y=y_pos_train)
    model_neg = dnn_model_train(nepoch=nb_epoch, nfeatures=n_feature, X=X_train, y=y_neg_train)
    model_neu = dnn_model_train(nepoch=nb_epoch, nfeatures=n_feature, X=X_train, y=y_neu_train)

    pos_train = model_pos.predict_proba(X_train, verbose = 0)
    pos_test = model_pos.predict_proba(X_test, verbose = 0)
    neg_train = model_neg.predict_proba(X_train, verbose = 0)
    neg_test = model_neg.predict_proba(X_test, verbose = 0)
    neu_train = model_neu.predict_proba(X_train, verbose = 0)
    neu_test = model_neu.predict_proba(X_test, verbose = 0)

    y_answer_train = [0]*len(X_pos_train) + [1]*len(X_neg_train)+ [2]*len(X_neu_train)
    y_answer_test = [0]*len(X_pos_test) + [1]*len(X_neg_test)+ [2]*len(X_neu_test)

    train_sentiment_list = []
    for i in xrange(len(X_train)):
        train_sentiment_list.append((pos_train[i][0],neg_train[i][0],neu_train[i][0]))
    test_sentiment_list = []
    for i in xrange(len(X_test)):
        test_sentiment_list.append((pos_test[i][0],neg_test[i][0],neu_test[i][0]))

    err_index_train = compute_error_rate(y_answer_train, train_sentiment_list)
    err_index_test = compute_error_rate(y_answer_test, test_sentiment_list)

    '''
    print len(X_train_pca)
    correct_pts = [i for j, i in enumerate(X_train_pca) if j not in err_index_train]
    print len(correct_pts)
    wrong_pts = [i for j, i in enumerate(X_train_pca) if j in err_index_train]
    print len(wrong_pts)
    '''
    
    print len(X_test_pca)
    correct_pts = [i for j, i in enumerate(X_test_pca) if j not in err_index_test]
    print len(correct_pts)
    wrong_pts = [i for j, i in enumerate(X_test_pca) if j in err_index_test]
    print len(wrong_pts)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(zip(*correct_pts)[0], zip(*correct_pts)[1], zip(*correct_pts)[2], c='r')
    if len(wrong_pts) > 0:
        ax.scatter(zip(*wrong_pts)[0], zip(*wrong_pts)[1], zip(*wrong_pts)[2], c='g')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
if __name__ == "__main__":
    main()