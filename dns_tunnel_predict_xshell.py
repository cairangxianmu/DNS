# -*- coding: utf-8 -*-
from dns_tunnel_train_model import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

def roc(y_test, predictions):
    y_true = []
    y_test = y_test.tolist()
    for i in y_test:
        if i.index(1)==2:
            y_true.append(1)
        elif i.index(1)==1:
            y_true.append(0)
        else:
            y_true.append(i.index(0))
    print("label",y_true)
    y_score = []
    for p in predictions:
        if p[0]==max(p):
            y_score.append(0)
        if p[1]==max(p):
            y_score.append(0)
        if p[2]==max(p):
            y_score.append(1)
    print("score",y_score)
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
    data_path = "./xshell_data"
    black_data = []
    for path in iterbrowse(data_path):
        with open(path) as f:
            for line in f:
                mdomain, subdomain = metadata2_domain_data(line)
                if subdomain is not None:
                    black_data.append(subdomain)
    return black_data


org_X = []

def get_xshell_data():
    # 子域名
    global org_X
    org_X = get_predict_data()
    labels = [LABEL.black]*len(org_X)

    volcab_file = "volcab.pkl"
    assert os.path.exists(volcab_file)
    pkl_file = open(volcab_file, 'rb')
    data = pickle.load(pkl_file)
    valid_chars, maxlen, max_features = data["valid_chars"], data["max_len"], data["volcab_size"]

    # Convert characters to int and pad
    X = [[valid_chars[y] if y in valid_chars else 0 for y in x] for x in org_X]
    X = pad_sequences(X, maxlen=maxlen, value=0.)

    # Convert labels to 0-1
    Y = to_categorical(labels, nb_classes=3)
    return X, Y, maxlen, max_features


def run():
    testX, testY, max_len, volcab_size = get_xshell_data()
    print ("X len:", len(testX), "Y len:", len(testY))
    print (testX[-1:])
    print (testY[-1:])

    model = get_cnn_model(max_len, volcab_size)

    filename = 'result/finalized_model.tflearn'
    loaded_model = model.load(filename)

    predictions = model.predict(testX)
    
    cnt = 0
    global org_X
    for i,p in enumerate(predictions):
        #if abs(p[2]-testY[i][2]) < 0.1:
        if p[2]>p[1] and p[2]>p[0]:
            cnt += 1
        else:
            print ("found data not detected:")
            print ("original subdomain:", org_X[i])
            print ("prediction compare:", p, testY[i])
    print ("Dectected cnt:", cnt, "total:", len(predictions))
    print ("Dectect Rate is:", cnt/(len(predictions)+.0))
    #p[0]:white p[1]:cnt p[2]:black
    roc(testY,predictions)

if __name__ == "__main__":
    run()
