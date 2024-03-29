# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

import tldextract
import tflearn
import os
import pickle
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import re

pattern = '[a-zA-Z]{2,62}\.[a-zA-Z]{2,62}'

data_path = "./test"
tag = "dns"
black_data, cdn_data, white_data = [], [], []
i = 0
# for path in iterbrowse(data_path):
#     #print(path)
#     with open(path,'rb') as f:
#         for line in f:
#             line = line.decode('gb18030')
#             data = line.split('|')
#             for s in data:
#                 if re.search(pattern, s):
#                     print(s)
#                     i = i + 1


class LABEL(object):
    white = 0
    cdn = 1
    black = 2


def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)

def get_local_data(tag="dns"):
    data_path = "./test"
    black_data, cdn_data, white_data = [], [], []
    for i in black_data:
        print(i)
    for path in iterbrowse(data_path):
        with open(path,'rb') as f:
                for line in f:
                    line = line.decode('gb18030')
                    data = line.split('|')
                    for s in data:
                        if re.search(pattern, s):
                            print(s)
                            mdomain, subdomain = metadata2_domain_data(s.strip())
                            if subdomain is not None:
                                if "white" in path:
                                    white_data.append(subdomain)
                                elif "cdn" in path:
                                    cdn_data.append(subdomain)
                                elif "black" in path:
                                    black_data.append(subdomain)
                                else:
                                    pass
    return black_data, cdn_data, white_data

def extract_domain(domain):
    suffix = {'.com','.la','.io', '.co', '.cn','.info', '.net', '.org','.me', '.mobi', '.us', '.biz', '.xxx', '.ca', '.co.jp', '.com.cn', '.net.cn', '.org.cn', '.mx','.tv', '.ws', '.ag', '.com.ag', '.net.ag', '.org.ag','.am','.asia', '.at', '.be', '.com.br', '.net.br', '.name', '.live', '.news', '.bz', '.tech', '.pub', '.wang', '.space', '.top', '.xin', '.social', '.date', '.site', '.red', '.studio', '.link', '.online', '.help', '.kr', '.club', '.com.bz', '.net.bz', '.cc', '.band', '.market', '.com.co', '.net.co', '.nom.co', '.lawyer', '.de', '.es', '.com.es', '.nom.es', '.org.es', '.eu', '.wiki', '.design', '.software', '.fm', '.fr', '.gs', '.in', '.co.in', '.firm.in', '.gen.in', '.ind.in', '.net.in', '.org.in', '.it', '.jobs', '.jp', '.ms', '.com.mx', '.nl','.nu','.co.nz','.net.nz', '.org.nz', '.se', '.tc', '.tk', '.tw', '.com.tw', '.idv.tw', '.org.tw', '.hk', '.co.uk', '.me.uk', '.org.uk', '.vg'}

    domain = domain.lower()
    names = domain.split(".")
    if len(names) >= 3:
        if ("."+".".join(names[-2:])) in suffix:
            return ".".join(names[-3:]), ".".join(names[:-3])
        elif ("."+names[-1]) in suffix:
            return ".".join(names[-2:]), ".".join(names[:-2])
    pos = domain.rfind("/")
    if pos >= 0: # maybe subdomain contains /, for dns tunnel tool
        ext = tldextract.extract(domain[pos+1:])
        subdomain = domain[:pos+1] + ext.subdomain
    else:
        ext = tldextract.extract(domain)
        subdomain = ext.subdomain
    if ext.suffix:
        mdomain = ext.domain + "." + ext.suffix
    else:
        mdomain = ext.domain
    return mdomain, subdomain


def metadata2_domain_data(log):
    mdomain, subdomain = extract_domain(log)
    return (mdomain, subdomain)


def get_data():
    black_x, cdn_x, white_x = get_local_data()
    black_y, cdn_y, white_y = [LABEL.black] * len(black_x), [LABEL.cdn] * len(cdn_x), [LABEL.white] * len(white_x)

    X = black_x + cdn_x + white_x
    labels = black_y + cdn_y + white_y
    print(X)
    rui = set(''.join(X))
    # Generate a dictionary of valid characters
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1
    print("max_features:", max_features)
    maxlen = np.max([len(x) for x in X])
    print("max_len:", maxlen)
    maxlen = min(maxlen, 256)

    # Z = [['a','b','c'],['d']]
    # valid_a = {a:idx+1 for idx,a in enumerate(set(''.join(['a','b','c','d','e'])))}
    # z = [[valid_a[y] for y in x] for x in Z]
    # dddd = valid_a['a']
    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = pad_sequences(X, maxlen=maxlen, value=0.)

    # Convert labels to 0-1
    Y = to_categorical(labels, nb_classes=3)

    volcab_file = "volcab.pkl"
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
    net = tflearn.lstm(net, 64, dropout=0.8)
    net = tflearn.fully_connected(net, 3, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
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
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model


def run():
    X, Y, max_len, volcab_size = get_data()

    print("X len:", len(X), "Y len:", len(Y))
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(trainX[:1])
    print(trainY[:1])
    print(testX[-1:])
    print(testY[-1:])

    model = get_cnn_model(max_len, volcab_size)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)

    filename = 'finalized_model.tflearn'
    model.save(filename)

    model.load(filename)
    print("Just review 3 sample data test result:")
    result = model.predict(testX[0:3])
    print(result)


if __name__ == "__main__":
    run()
