#-*- coding:utf8 -*-

import jieba
import jieba.analyse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

jieba.analyse.set_stop_words('stop_words.txt')
jieba.load_userdict('userdict.txt')
jieba.enable_parallel()

#kmeans
key_list = []
with open('sms_0000.csv', 'r') as f:
    sms = f.readlines()[:10000]
    for sm in sms:
        keywords = jieba.analyse.extract_tags(sm, topK=5, withWeight=True, allowPOS=('n','vn','v'))
        keys = []
        for item in keywords:
            keys.append(item[1])
        while len(keys) < 5:
            keys.append(0)
        key_list.append(keys[:5])
train = pd.DataFrame(key_list)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

#train = pd.DataFrame(key_list)
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
sc_scores = []
meandistortions = []
for t in clusters:
    kmeans = KMeans(n_clusters=t)
    kmeans_model = kmeans.fit(train)
    sc_score = silhouette_score(train, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    meandistortions.append(sum(np.min(cdist(train, kmeans_model.cluster_centers_, 'euclidean'), axis=1))/train.shape[0])

    #可视化结果
    YY = kmeans_model.labels_
    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111, 2)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(YY[i]),
                     color=plt.cm.Set1(YY[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(train)
    plot_embedding(X_tsne)
    plt.show()

#轮廓系数和肘部观测
plt.plot(clusters, sc_scores, marker='*')
plt.xticks(clusters)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhpouette Coefficient Score')
plt.show()

plt.plot(clusters, meandistortions, marker='o')
plt.xticks(clusters)
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()

# 保存预测结果
with open('sms_0000.csv', 'r') as f:
    sms = f.readlines()[:10000]
    label_dict = {1:[u'签收'], 2:[u'上新', u'新品上架'], 3:[u'快递', u'速递', u'速运', u'查收'], 4:[u'双11',u'双十一'], 5:[u'成功付款'], 6:[u'没有付款', u'抓紧时间付款', u'点击付款', u'还未付款']}

    with open('sms_0000_label.csv', 'w') as cf:
        for sm in sms:
            keywords = jieba.analyse.extract_tags(sm, topK=3, withWeight=False, allowPOS=('n', 'vn', 'v'))
            tmp = 0
            for item in keywords:
                if tmp == 0:
                    for i in range(1, 7, 1):
                        if item in label_dict[i]:
                            cf.write(sm.strip() + ',' + str(i) + '\n')
                            tmp = i
                            break
                if tmp != 0:
                    break
            if tmp == 0:
                cf.write(sm.strip() + ',' + str(0) + '\n')