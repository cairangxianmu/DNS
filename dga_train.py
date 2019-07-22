# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

import os
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import csv
from common import *


def normalization(x):
    Min = np.max(x)
    Max = np.min(x)
    x = (x - Min) / (Max - Min)
    return np.round(x, decimals=5)


def process(X):
    feature = []
    for index, i in enumerate(X):
        # , getyuanyin(word[index])
        feature.append([getrootclass(i), getlen(i), getshan(i)])
    feature = np.array(feature)
    feature[:, 1] = normalization(feature[:, 1])
    feature[:, 2] = normalization(feature[:, 2])
    # feature[:, 3] = normalization(feature[:, 3])
    feature = feature.tolist()
    print(feature)

    return feature


def get_local_data(tag="train"):
    data_path = "DGA_sample"
    black_data, white_data = [], []
    for dir_path in os.listdir(data_path):
        if ("black", "white") and tag in dir_path:
            path = data_path + "/" + dir_path
            print(path)
            with open(path) as f:
                for line in f:
                    subdomain = line.replace('\n', '').replace('\t', '').rstrip('.')
                    if subdomain is not None:
                        if "white" in path:
                            white_data.append(subdomain)
                        elif "black" in path:
                            black_data.append(subdomain)
                        else:
                            pass
                            # print ("pass path:", path)
                    # else:
                    #    print ("unknown line:", line, " in file:", path)
    return black_data, white_data


class LABEL(object):
    white = 0
    black = 1


def get_data():
    test_path = "dga_predict/dga_sample.csv"
    black_x, white_x = get_local_data()
    black_y, white_y = [LABEL.black] * len(black_x), [LABEL.white] * len(white_x)

    X = black_x + white_x
    labels = black_y + white_y

    X, testX, labels, testY = train_test_split(X, labels, test_size=0.2, random_state=42)
    # rui = set(''.join(X))
    with open(test_path,"w",encoding="ASCII") as f:
        writer = csv.writer(f)
        for i,pre in enumerate(testX):
            # print(pre)
            writer.writerow([pre]+[str(testY[i])])

    # Generate a dictionary of valid characters
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1
    print("max_features:", max_features)
    maxlen = np.max([len(x) for x in X])
    print("max_len:", maxlen)
    maxlen = min(maxlen, 256)

    X = [[valid_chars[y] for y in x] for x in X]
    feature = process(X)
    for index, i in enumerate(feature):
        i.extend(X[index])
    # print(feature)
    X = pad_sequences(feature, maxlen=maxlen, dtype='float32', value=0.)
    # Convert labels to 0-1
    Y = to_categorical(labels, nb_classes=2)

    volcab_file = "volcab_dga.pkl"
    output = open(volcab_file, 'wb')
    # Pickle dictionary using protocol 0.
    data = {"valid_chars": valid_chars, "max_len": maxlen, "volcab_size": max_features}
    pickle.dump(data, output)
    output.close()

    return X, Y, maxlen, max_features


def get_rnn_model(max_len, volcab_size):
    # Network building
    net = tflearn.input_data([None, max_len])
    net = tflearn.embedding(net, input_dim=volcab_size, output_dim=64)
    net = tflearn.lstm(net, 64, dropout=0.2)
    net = tflearn.fully_connected(net, 2, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='binary_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=3)
    return model


def get_cnn_model(max_len, volcab_size):
    # Building convolutional network
    network = tflearn.input_data(shape=[None, max_len], name='input')
    network = tflearn.embedding(network, input_dim=volcab_size, output_dim=64)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.2)
    network = fully_connected(network, 2, activation='sigmoid')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='binary_crossentropy', name='target')
    model = tflearn.DNN(network, tensorboard_verbose=3)
    return model


def run():
    X, Y, max_len, volcab_size = get_data()

    print("X len:", len(X), "Y len:", len(Y))
    # trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)


    model = get_cnn_model(max_len, volcab_size)
    model.fit(X, Y, n_epoch=60, shuffle=True, validation_set=(X, Y), show_metric=True, batch_size=64)

    filename = 'result_dga/finalized_model.tflearn'
    model.save(filename)

    model.load(filename)
    print("Just review 3 sample data test result:")
    # result = model.predict(testX[0:10])
    # print(result, ":", trainY[0:10])


if __name__ == "__main__":
    run()
