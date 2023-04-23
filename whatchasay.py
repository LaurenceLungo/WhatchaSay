import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jieba as jb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from argparse import ArgumentParser
from time import time
from hanziconv import HanziConv
from sklearn.metrics import silhouette_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def pre_process(text):
    text = HanziConv.toTraditional(text)

    # load cantonese corpus
    # jb.load_userdict('util/dict/canto_dict.txt')
    vocabs = list(jb.cut(text))
    pp_text = " ".join(vocabs)
    return pp_text

@ignore_warnings(category=ConvergenceWarning)
def find_opt_k(dat, feat):
    K = list(np.linspace(2, dat.size, 50, dtype=int))
    score_list = []
    k_list = []
    max_silhouette = [-1, 0]
    dec_trend = 0

    for k in K:
        clusterer = KMeans(n_clusters=k, random_state=0, n_init='auto')
        preds = clusterer.fit_predict(feat)
        score = silhouette_score(feat, preds)
        score_list.append(score)
        k_list.append(k)
        print("For k = {}, silhouette score is {}".format(k, score))
        if score > max_silhouette[0]:
            max_silhouette = [score, k]
            dec_trend = 0
        else:
            dec_trend += 1
            if dec_trend == 10:
                break
    print("--- Found optimal k:", max_silhouette[1], "---\n")
    return max_silhouette[1]


start_time = time()

parser = ArgumentParser(description='Clustering text.')
parser.add_argument('dataset', metavar='d', nargs='?', help='directory to training set')
parser.add_argument('test', metavar='t', nargs='?', help='test phrase')
args = parser.parse_args()
args_value = vars(args)

tdf = pd.read_csv(args_value['dataset'])
stopwords = [' ', '\n', 'previous', 'message']
vec = TfidfVectorizer(stop_words=stopwords, preprocessor=pre_process)
vec.fit(tdf.values)
for (k, v) in vec.vocabulary_.items():
    if len(k) == 1:
        print(k, ": ", v)
features = vec.transform(tdf.values)

cls = KMeans(n_clusters=find_opt_k(tdf, features), random_state=0, n_init='auto')
cls.fit(features)
prediction = cls.predict(vec.transform([pre_process(args_value["test"])]))

output_str = "\"" + args_value['test'] + "\"/"
for i in range(len(cls.labels_)):
    if cls.labels_[i] == prediction:
        output_str += ("\"" + tdf.values[i][0] + "\"/")
output_str = output_str.replace('\n', ' ')
file = open('output.txt', 'w')
file.write(output_str)
file.close()

print("Output:", output_str)
print("Running time:", time() - start_time, "seconds")


pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(features.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')
# plt.show()
plt.savefig('clusters.png')
