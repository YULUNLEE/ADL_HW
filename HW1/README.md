# Applid Deep Learning HW1
## step 1: environment setup

* pip install -r requirements.txt (The TA sample), but you have to delete the torch (cpu version) 
* pytorch-gpu version
* pandas

thus you need to add two commands below:
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install pandas
```

## step 2: data download
```shell
bash ./download.sh
```

## step 3: implement the intent model
#### completed training procedure:
"${1}": path to the testing file.

"${2}": path to the output predictions.

ex : bash ./intent_cls.sh ./data/intent/test.json ./intent_pred.csv
```shell
bash ./intent_cls.sh "${1}" "${2}"
```
#### reproduce the intent best kaggle submission:
"${1}": path to the testing file.

"${2}": path to the output predictions.

"${3}": path to the best intent model pkl file.

ex : bash ./intent_bestKaggle.sh ./data/intent/test.json ./best_intent_pred.csv ./intent_best91.33.pkl
```shell
bash ./intent_bestKaggle.sh "${1}" "${2}" "${3}"
```
## step 4: implement the slot_tags model
#### completed training procedure:
"${1}": path to the testing file.

"${2}": path to the output predictions.

ex : bash ./slot_tag.sh ./data/slot/test.json ./slot_pred.csv
```shell
bash ./slot_tag.sh "${1}" "${2}"
```
#### reproduce the slot best kaggle submission:
"${1}": path to the testing file.

"${2}": path to the output predictions.

"${3}": path to the best slot model pkl file.

ex : bash ./tag_bestKaggle.sh ./data/slot/test.json ./best_slot_pred.csv ./tagging_best82.252.pkl
```shell
bash ./tag_bestKaggle.sh "${1}" "${2}" "${3}"
```
