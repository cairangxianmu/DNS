import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import numpy as np

def roc(y_test, predictions):
    y_true = []
    # y_test = y_test.tolist()
    for i in y_test:
        if i.index(1)==2:
            y_true.append(1)
        else:
            y_true.append(i.index(1))
    print("label",y_true)
    y_score = []
    for p in predictions:
        if p[0]==max(p):
            y_score.append(0)
        if p[1]==max(p):
            y_score.append(1)
        if p[2]==max(p):
            y_score.append(1)
    print(y_score)
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

if __name__ == "__main__":
    y_test = [0,1,1,1,1,1,0,0,0,1,0]
    predictions = [0,1,1,1,1,0,0,0,0,1,1]
    roc(y_test,predictions)