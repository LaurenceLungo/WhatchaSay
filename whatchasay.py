import gensim
import jieba
import pandas as pd
import numpy as np
from opencc import OpenCC


def read_corpus(para):
    # with smart_open.open(fname) as f:
    #     for i, line in enumerate(f):
    #         tokens = gensim.utils.simple_preprocess(line)
    #         yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
    c_para = para.copy()
    for i in range(len(c_para)):
        c_para[i] = c_para[i].replace("(previous message:", "")
        c_para[i] = c_para[i].replace(")", "")
        c_para[i] = c_para[i].replace('\n', " ")
        c_para[i] = ' '.join(jieba.cut(c_para[i]))
    # print(para)
    for j, line in enumerate(c_para):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [j])


def transform_enquiry(text):
    text = text.replace("(previous message:", "")
    text = text.replace(")", "")
    text = text.replace('\n', " ")
    # jieba.load_userdict("./util/dict/dict.txt")
    text = ' '.join(jieba.cut(text))
    print(text)
    cc = OpenCC('s2hk')
    text = cc.convert(text)
    print(gensim.utils.simple_preprocess(text))
    return gensim.utils.simple_preprocess(text)


para = pd.read_csv('dataset.csv').Enquiry.values
for k in range(len(para)):
    line = para[0]
    np.delete(para, 0)
    split_line = line.split('\n')
    para = np.append(para, split_line)
train_corpus = list(read_corpus(para))
# print(train_corpus)

model = gensim.models.doc2vec.Doc2Vec(vector_size=50)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
vector = model.infer_vector(transform_enquiry('點樣收費?'))
sims = model.docvecs.most_similar([vector])

print(sims)
print()

for index, conf in sims:
    print(para[index])
    print('--------------------------')