import pandas as pd
import matplotlib.pyplot as plt
import jieba as jb
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from argparse import ArgumentParser
from opencc import OpenCC
from time import time
from hanziconv import HanziConv
from sklearn.metrics import silhouette_score
import numpy as np


def pre_process(text):
    # cc = OpenCC('s2hk')
    # text = cc.convert(text)
    text = HanziConv.toTraditional(text)
    vocabs = list(jb.cut(text))
    pp_text = " ".join(vocabs)
    return pp_text


ini = time()
stopwords = [' ', '\n', 'previous', 'message']
random_state = 0

parser = ArgumentParser(description='Clustering text.')
parser.add_argument('dataset', metavar='d', nargs='?',
                    help='directory to training set')
parser.add_argument('test', metavar='t', nargs='?',
                    help='test phrase')
args = parser.parse_args()
args_value = vars(args)

tdf = pd.read_csv(args_value['dataset'])
# for k in range(len(tdf)):
#     line = str(tdf.iloc[[0]].values[0][0])
#     # print('line:', line)
#     tdf = tdf.drop([0])
#     tdf = tdf.reset_index(drop=True)
#     # print(tdf)
#     split_line = pd.DataFrame(line.split('\n'), columns={'Enquiry'})
#     # print(type(split_line))
#     tdf = tdf.append(split_line)


vec = TfidfVectorizer(stop_words=stopwords, preprocessor=pre_process)
vec.fit(tdf.Enquiry.values)
for (k, v) in vec.vocabulary_.items():
    if len(k) == 1:
        print(k, ": ", v)
features = vec.transform(tdf.Enquiry.values)

distortions = []
K = list(np.linspace(200, tdf.size, 10, dtype=int))
# for k in K:
#     cls = KMeans(n_clusters=k, random_state=random_state)
#     cls.fit(features)
#     distortions.append(cls.inertia_)
#     print('k=', k, 'done')
# plt.figure(figsize=(16, 8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

s_score = []
kk=[]
max_silhouette = [-1, 0]
dec_trend = 0
for k in K:
    clusterer = KMeans(n_clusters=k, random_state=random_state)
    preds = clusterer.fit_predict(features)
    centers = clusterer.cluster_centers_
    score = silhouette_score(features, preds)
    s_score.append(score)
    kk.append(k)
    print("For n_clusters = {}, silhouette score is {})".format(k, score))
    if score > max_silhouette[0]:
        max_silhouette = [score, k]
    else:
        dec_trend += 1
        if dec_trend == 3:
            print("found optimal k:", max_silhouette[1])
            break
plt.figure(figsize=(16, 8))
plt.plot(kk, s_score, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette')
plt.show()

cls = KMeans(n_clusters=max_silhouette[1], random_state=random_state)
cls.fit(features)
print('pp_text:', pre_process(args_value["test"]))
prediction = cls.predict(vec.transform([pre_process(args_value["test"])]))
print(prediction)
for i in range(len(cls.labels_)):
    if cls.labels_[i] == prediction:
        print(tdf.Enquiry.values[i])
        print("------------------------")

# pca = PCA(n_components=2, random_state=random_state)
# reduced_features = pca.fit_transform(features.toarray())

# # reduce the cluster centers to 2D
# reduced_cluster_centers = pca.transform(cls.cluster_centers_)
#
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cls.predict(features))
# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')
# plt.show()

print("cost:", time() - ini, "seconds")
