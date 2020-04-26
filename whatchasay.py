import pandas as pd
import matplotlib.pyplot as plt
import jieba as jb
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from argparse import ArgumentParser
from opencc import OpenCC
from time import time


def pre_process(text):
    cc = OpenCC('s2hk')
    text = cc.convert(text)
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

vec = TfidfVectorizer(stop_words=stopwords, preprocessor=pre_process)
vec.fit(tdf.Enquiry.values)
for (k, v) in vec.vocabulary_.items():
    if len(k) == 1:
        print(k, ": ", v)
features = vec.transform(tdf.Enquiry.values)

cls = KMeans(n_clusters=80, random_state=random_state)
cls.fit(features)

prediction = cls.predict(vec.transform([args_value["test"]]))
print(prediction)
for i in range(len(cls.labels_)):
    if cls.labels_[i] == prediction:
        print(tdf.Enquiry.values[i])
        print("------------------------")

pca = PCA(n_components=2, random_state=random_state)
reduced_features = pca.fit_transform(features.toarray())

# # reduce the cluster centers to 2D
# reduced_cluster_centers = pca.transform(cls.cluster_centers_)
#
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cls.predict(features))
# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')
# plt.show()

print("cost:", time() - ini, "seconds")
