# -*- coding: utf-8 -*-
from dga_train import *
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score

def roc(y_test, predictions):
    y_true = []
    y_test = y_test.tolist()
    for i in y_test:
        if i.index(1) == 1:
            y_true.append(1)
        else:
            y_true.append(0)
    print("label", y_true)
    y_score = []
    result = []
    for p in predictions:
        y_score.append(p[1])
        if p[0] == max(p):
            result.append(0)
        if p[1] == max(p):
            result.append(1)
    print("predict", result)
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == result[i]:
            count += 1
    accuracy = count/len(y_true)

    fpr, tpr, threshold = roc_curve(y_true, y_score)  #计算真正率和假正率
    precision = precision_score(y_true, result, average='binary')
    recall = recall_score(y_true, result, average='binary')
    f1 = f1_score(y_true, result, average='micro')
    roc_auc = auc(fpr, tpr)  #计算auc的值
    print("precision", precision, "recall", recall, "accuracy=", accuracy, "roc_auc=", roc_auc, "F1=", f1)

def get_predict_data():
    data_path = "xshell_data/"
    # data_path = "test/"
    black_data, white_data = [], []
    for dir_path in os.listdir(data_path):
        if "black" or "white" in dir_path:
            path = data_path + "/" + dir_path
            print(path)
            with open(path) as f:
                for line in f:
                    # domain, label = line.split(',')
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
        else:
            pass
    return black_data, white_data

def get_xshell_data():
    # 子域名
    global org
    black_x,white_x = get_predict_data()
    black_y, white_y = [LABEL.black]*len(black_x), [LABEL.white]*len(white_x)
    org = black_x + white_x
    labels = black_y + white_y

    volcab_file = "volcab_dga.pkl"
    assert os.path.exists(volcab_file)
    pkl_file = open(volcab_file, 'rb')
    data = pickle.load(pkl_file)
    valid_chars, maxlen, max_features = data["valid_chars"], data["max_len"], data["volcab_size"]

    # Convert characters to int and pad
    X = [[valid_chars[y] if y in valid_chars else 0 for y in x] for x in org]
    feature = process(X)
    for index, i in enumerate(feature):
        i.extend(X[index])
    X = pad_sequences(feature, maxlen=maxlen, value=0.)

    # Convert labels to 0-1
    Y = to_categorical(labels, nb_classes=2)
    return X, Y, maxlen, max_features


def run():
    testX, testY, max_len, volcab_size = get_xshell_data()
    print("X len:", len(testX), "Y len:", len(testY))
    print(testX[-1:])
    print(testY[-1:])

    model = get_cnn_model(max_len, volcab_size)

    filename = 'result_dga/finalized_model.tflearn'
    model.load(filename)

    predictions = model.predict(testX)

    global org
    for i, p in enumerate(predictions):
        if p[0] > p[1]:
            print("found white data:",org[i])
        elif p[0] < p[1]:
            print("found black data:",org[i])
        print("prediction compare:", p, testY[i])
    # p[0]:white p[1]:black
    roc(testY, predictions)


if __name__ == "__main__":
    run()
