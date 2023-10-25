# WhatchaSay
This is an exercise to cluster Chinese dialogs by their semantic meaning.

Given a **dataset of Chinese sentences** and a **query string**, the program would retrieve the all the sentences that are relevant to the query from the dataset.
<br>
This project runs on Python3.

## Approach
The program is optimized for chinese dialogs.\
<br>
It first preprocesses the sentences:
+ a Chinese corpus is used to split the continuous sentence into a list of vocabularies.
+ simplified Chinese characters (if any) are converted to traditional characters.
+ single-character vocabularies are discarded as they are not intent-deterministic. 
+ special characters like punctuations, whitespaces and newlines are discarded.
<br>

It then vectorizes the sentence and generates features from it.\
<br>
Then, the features are fed to a **K-Means** clustering model. The number of cluster k is auto-determined by the **silhouette method**.\
<br>
Finally when a query string is given, the program predicts the cluster it belongs to and return all members in the same cluster.

## Findings
I tried to load a Cantonese corpus to Jieba (the Chinese vocab splitter) for better handling of Cantonese. However, it makes the running time significantly longer. Plus there is no significant improvement on the clustering accuracy. One possible explaination is that the Cantonese corpus makes the splitter create a lot of single-character vocabularies which are not intent-deterministic.

## Future improvements
To increase clustering accuracy for English, stemming technique can be applied.

To further increase clustering accuracy for Chinese, the Chinese Corpus can be used to determine the part of speech of vocabularies. Verbs and nouns should be more heavily weighted as a feature beacuse they are a stronger indicater of dialog intent compared to other part of speeches.

## Dependencies
+ matplotlib
+ numpy
+ pandas
+ sklearn
+ jieba
+ hanziconv

## Installation
Install all the dependencies from Pypi:
```sh
$ pip install -r requirements.txt
```

## Usage
### Quick Start
To use the program to query the toy dataset with「請問點收費？」, run
```sh
$ python whatchasay.py dataset/text_msg_mini_dataset.csv 請問收費？

# Building prefix dict from the default dictionary ...
# Loading model from cache /var/folders/7l/vp59r70s4kbg9vmm4kjd_tlr0000gn/T/jieba.cache
# Loading model cost 0.283 seconds.
# Prefix dict has been built successfully.
# For k = 2, silhouette score is 0.01740093422452308
# For k = 3, silhouette score is 0.020572319002769426
# For k = 4, silhouette score is 0.029639337825562884
# For k = 5, silhouette score is 0.02642755666482454
# For k = 7, silhouette score is 0.03581599385276004
# For k = 8, silhouette score is 0.031982013004888135
# For k = 9, silhouette score is 0.035133752653544854
# For k = 10, silhouette score is 0.033947044983219064
# For k = 12, silhouette score is 0.03649655605349418
# For k = 13, silhouette score is 0.03838591859192241
# For k = 14, silhouette score is 0.040814509890555226
# For k = 15, silhouette score is 0.04519756158889073
# For k = 17, silhouette score is 0.04171362758342194
# For k = 18, silhouette score is 0.039736352848211705
# For k = 19, silhouette score is 0.03653667163587451
# For k = 20, silhouette score is 0.04015206927645096
# For k = 22, silhouette score is 0.05644952470906704
# For k = 23, silhouette score is 0.05514351617328082
# For k = 24, silhouette score is 0.05585357287181582
# For k = 26, silhouette score is 0.05638831397274753
# For k = 27, silhouette score is 0.0535216626481989
# For k = 28, silhouette score is 0.053267663541572335
# For k = 29, silhouette score is 0.05213170976983336
# For k = 31, silhouette score is 0.051260758441377896
# For k = 32, silhouette score is 0.0516992934371453
# For k = 33, silhouette score is 0.053168947993170346
# For k = 34, silhouette score is 0.05152934990087739
# --- Found optimal k: 22 ---

# Output: "請問收費？"/"請問收費方式是怎樣的？"/"收費方式有哪些？"/
```

### Program usage
To execute the program, run
```sh
$ python whatchasay.py <text-dataset-dir> <query-string>
```
The matching results will be stored in a newly created **output.txt** in the same directory.\
The clusters plot will be stored in a newly created **clusters.png** in the same directory.\
<br>


To see more information about the usage, run

```sh
$ python whatchasay.py -h
```

### Input dataset csv format
The program takes a dataset formatted as follows:

```sh
"收費如何？"
"請問我係咪抽唔中？"
"I want to book a table"
```

A sample text message dataset can be found in ```dataset/text_msg_mini_dataset.csv``` for quick experiements

### Output format
The program creates an output file formatted as follows:
```sh
"收費點樣？"/"收费如何？"/"收費係點？"/"如何收費?係咪有優惠碼？"/"收费如何？"/
```
The first dialog is the input query string, while those following it are the dialogs of similar meanings from the dataset.
