from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.externals import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
import json
import jieba
import numpy as np
import random


titleall = []
classnum = 3
numall = 0
def loadDataset():

    with open('Boy-Girl-3356-3358.json', encoding = 'utf-8-sig') as data_file:    
        data = json.load(data_file, encoding = 'utf-8-sig')
    with open('HBL-406-407.json', encoding = 'utf-8-sig') as data_file:    
        data2 = json.load(data_file, encoding = 'utf-8-sig')
    with open('Elephants-3304-3306.json', encoding = 'utf-8-sig') as data_file:    
        data3 = json.load(data_file, encoding = 'utf-8-sig')
    datalist = data["articles"]
    d2list = data2["articles"]
    d3list = data3["articles"]
    datalist += d2list
    datalist += d3list
    num = len(datalist)
    numall = num
    print(num)
    dataset = []
    #for i in range(num):
    for i in range(num):
        datatmp = datalist[i]
        tmp = ""
        if datatmp["article_title"] == "":
            continue
        
        if "content" in datatmp:
            article = datalist[i]["content"]
            #print(data["articles"][i]["article_title"])
            titleall.append(datalist[i]["article_title"])
            words = jieba.cut(article, cut_all=False)
            for word in words:
                tmp += word
                tmp += " "
            dataset.append(tmp)
        
        else:continue
    print(len(dataset))
    return dataset

def transform(dataset,n_features=1000):
    #vectorizer = joblib.load('tfidf_fit_result.pkl')
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, ngram_range=(1,1), min_df=2,use_idf=True)
    #vectorizer = CountVectorizer(max_df=0.5, max_features=n_features,ngram_range=(1,1), min_df=2)

    X = vectorizer.fit_transform(dataset)
    #joblib.dump(vectorizer, 'tfidf_fit_result.pkl')
    return X,vectorizer

def train(X,vectorizer,true_k=10,minibatch = False,showLable = False):  
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=1,
                    verbose=False)
    #km = joblib.load('km_cluster_fit_result.pkl')
    km.fit(X)    
    #joblib.dump(km, 'km_cluster_fit_result.pkl')
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    
    cluster_labels = km.labels_
    #print("分群結果：")
    #print(cluster_labels)
    #print("---")
    
    for j in range(len(cluster_labels)):
        print(str(cluster_labels[j])+" : "+titleall[j])

    
    print("----------------------------")
    '''
    for m in range(classnum):
        for j in range(len(cluster_labels)):
            if cluster_labels[j] == m:
                #print(titleall[j], end='')
                print(titleall[j])
        print("----------------------------")
    '''
    return -km.score(X)
    
    
def main():
    dataset = loadDataset()
    X,vectorizer = transform(dataset,n_features=300)
    train(X,vectorizer,true_k=classnum,showLable=True)/len(dataset)
    #print (score)

main()
