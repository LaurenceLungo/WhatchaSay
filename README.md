# WhatchaSay
This is a Python project which clusters a set of dialogs by their meaning.\
<br>
This project runs on Python3.

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

