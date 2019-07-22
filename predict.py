# -*- coding: utf-8 -*-
from dga_train import *
import csv

def result(predictions,path):
    result = []
    for p in predictions:
        if p[0] > p[1]:
            result.append(0)
        else:
            result.append(1)
    with open(path,"w",encoding="ASCII") as f:
        writer = csv.writer(f)
        for i,pre in enumerate(org):
            print(pre)
            writer.writerow([pre]+[str(result[i])])
    print("写入成功",path)

def get_predict_data(path):
    data = []
    with open(path) as f:
        for line in f:
            subdomain = line.replace('\n', '').replace('\t', '').rstrip('.')
            data.append(subdomain)
    return data

def get_xshell_data(volcab_file,data_path):
    # 子域名
    global org
    org = get_predict_data(data_path)
    labels = [LABEL.black]*len(org)
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

def main(data_path,out_path,filename):
    volcab_file = "volcab_dga.pkl"

    testX, testY, max_len, volcab_size = get_xshell_data(volcab_file, data_path)
    # print("X len:", len(testX), "Y len:", len(testY))
    # print(testX[-1:])
    # print(testY[-1:])
    model = get_cnn_model(max_len, volcab_size)
    model.load(filename)
    predictions = model.predict(testX)
    result(predictions,out_path)

def run():
    dga_path = "dga_predict/dga_sample.csv"
    dns_path = "dga_predict/dns_sample.csv"
    dga_out_path = "中央民族大学-68422-DGA-有监督.csv"
    dns_out_path = "中央民族大学-68422-DNS-有监督.csv"
    dga_filename = 'result_dga/finalized_model.tflearn'
    dns_filename = 'result_dns/finalized_model.tflearn'
    main(dga_path,dga_out_path,dga_filename)
    main(dns_path,dns_out_path,dns_filename)
    print("程序运行结束")


if __name__ == "__main__":
    run()
