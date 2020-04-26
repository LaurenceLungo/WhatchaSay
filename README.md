# WhatchaSay
This is a Python project which clusters dialogs by their meaning.\
<br>
This project runs on Python3.

## Dependencies
+ matplotlib 3.2.1
+ numpy 1.18.3
+ pandas 1.0.3
+ sklearn 0.0
+ jieba 0.42.1
+ hanziconv 0.3.2

## Installation
Install all the dependencies from Pypi:
```sh
$ pip install -r requirements.txt
```

## Usage
To execute the program, run
```sh
$ python whatchasay.py <input-dataset-directory> <test-phrase>
# example: python whatchasay.py enquiry_similarity_training_set.csv  請問收費？
```
The result will be stored in a newly created **output.txt** in the same directory.\
<br>
To visualize the clustering result, add the -v flag
```sh
$ python whatchasay.py <input-dataset-directory> <test-phrase> -v
# example: python whatchasay.py enquiry_similarity_training_set.csv  請問收費？ -v
```

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

### Output format
The program creates an output file formatted as follows:
```sh
"收費點樣？"/"收费如何？"/"收費係點？"/"如何收費?係咪有優惠碼？"/"收费如何？"/
```
The first dialog is the input test phrase, while those following it are the dialogs of the same meaning from the training set.

## Approach
The program is optimized for chinese dialogs.\
<br>
It first uses a Chinese corpus to split the continuous sentence into a list of vocabularies and converts simplified Chinese characters (if any) to traditional characters. Single-character vocabularies are discarded as they are not intent-deterministic.\
<br>
It then **vectorizes** the sentence and generates features from it.\
<br>
Then, the features are fed to a **K-Means** clustering model. The number of cluster k is auto-determined by the **silhouette method**.\
<br>
Finally when a test phrase is given, the program predicts the cluster it belongs to and return all members in the same cluster.

## Findings
I tried to load a Cantonese corpus to Jieba (the Chinese vocab splitter) for better handling of Cantonese. However, it makes the running time significantly longer. Plus there is no significant improvement on the clustering accuracy. One possible explaination is that the Cantonese corpus makes the splitter create a lot of single-character vocabularies which are not intent-deterministic.

## Future improvements
To increase clustering accuracy for English, stemming technique can be applied.

To further increase clustering accuracy for Chinese, the Chinese Corpus can be used to determine the part of speech of vocabularies. Verbs and nouns should be more heavily weighted as a feature beacuse they are a stronger indicater of dialog intent compared to other part of speeches.
