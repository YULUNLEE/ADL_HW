# Applied Deep Learning HW3  (NLG-Summarization)
## step 1: environment setup
* pip install -r requirements.txt 
* pytorch-gpu version

thus you need to use two commands below:
```shell
pip install -r requirements.txt
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
## step 2: data download
####I use wget command to download the datas on dropbox. 
If you don't have wget, plz download the wget with this address:
https://eternallybored.org/misc/wget/
```shell
bash download.sh
```

##step 3: fine-tune mt5-small
"${1}": path to the training data jsonl file.

"${2}": path to the prediction data jsonl file.

"${3}": path to the output submission jsonl file.

ex : bash run_summarization.sh ./data/train.jsonl ./data/public.jsonl ./output/submission.jsonl
```shell
bash run_summarization.sh "${1}" "${2}" "${3}"
```

##step 4: reproduced the best submission 
"${1}": path to the prediction data jsonl file.

"${2}": path to the output submission jsonl file.

ex : bash run.sh ./data/public.jsonl ./output/best_submission.jsonl

```shell
bash run.sh "${1}" "${2}"
```
## Replenishment
* if you didn't change any repository of file, you can directly use the example commands to complete the different tasks.
* Step 3 task might cost several hours, if you only want to reproduce the best submission file, just run step 4. 