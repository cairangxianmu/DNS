# -*- coding: utf-8 -*-
from dga_train import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.svm import SVC

def cal_rate(result, thres):
    all_number = len(result[0])
    # print all_number
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        disease = result[0][item]
        if disease >= thres:
            disease = 1
        if disease == 1:
            if result[1][item] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[1][item] == 0:
                TN += 1
            else:
                FN += 1
    # print TP+FP+TN+FN
    accracy = float(TP+FP) / float(all_number)
    if TP+FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP+FP)
    TPR = float(TP) / float(TP+FN)
    TNR = float(TN) / float(FP+TN)
    FNR = float(FN) / float(TP+FN)
    FPR = float(FP) / float(FP+TN)
    # print accracy, precision, TPR, TNR, FNR, FPR
    return accracy, precision, TPR, TNR, FNR, FPR

def roc1(label, predictions):
    prob = []
    for p in predictions:
        prob.append(p[0])
    # prob = prob.tolist()
    print(prob)
    label = label.tolist()
    threshold_vaule = sorted(prob)
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
    # calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob, label), threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR

    AUC = np.trapz(TPR_array, FPR_array)
    threshold = np.argmin(abs(FNR_array - FPR_array))
    EER = (FNR_array[threshold] + FPR_array[threshold]) / 2
    plt.plot(FPR_array, TPR_array)
    plt.title('roc')
    plt.xlabel('FPR_array')
    plt.ylabel('TPR_array')
    plt.show()

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
    for p in predictions:
        y_score = p[0]
        # if p[0] == max(p):
        #     y_score.append(0)
        # if p[1] == max(p):
        #     y_score.append(1)
    clf = SVC()
    y_score = clf.decision_function(2,predictions)
    print("score", y_score)
    fpr, tpr, threshold = roc_curve(y_true, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def get_predict_data():
    data_path = "xshell_data"
    black_data, white_data = [], []
    for dir_path in os.listdir(data_path):
        if "black" or "white" in dir_path:
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
    X = pad_sequences(X, maxlen=maxlen, value=0.)

    # Convert labels to 0-1
    Y = to_categorical(labels, nb_classes=2)
    return X, Y, maxlen, max_features


def run():
    testX, testY, max_len, volcab_size = get_xshell_data()
    print("X len:", len(testX), "Y len:", len(testY))
    print(testX[-1:])
    print(testY[-1:])

    model = get_cnn_model(max_len, volcab_size)

    filename = 'result/finalized_model.tflearn'
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
    roc1(testY, predictions)


if __name__ == "__main__":
    run()
