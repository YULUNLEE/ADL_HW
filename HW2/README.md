# Applied Deep Learning HW2 (Bert QA)
## step 1: environment setup

* pip install -r requirements.txt 
* pytorch-gpu version
* eval tokenizer (zh_core_web_md)

thus you need to use three commands below:
```shell
pip install -r requirements.txt
pip (conda) install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
python -m spacy download zh_core_web_md
```

## step 2: data download
```shell
bash download.sh
```

## step 3: create public predictions.json
#### implement evaluation procedure:
"${1}": path to the context file.

"${2}": path to the public file.

"${3}": path to the output public predictions.

ex : bash run.sh ./data/context.json ./data/public.json ./output/predictions_public.json
```shell
bash run.sh "${1}" "${2}" "${3}"
```

## step 4: create private predictions.json
#### implement evaluation procedure:
"${1}": path to the context file.

"${2}": path to the private file.

"${3}": path to the output private predictions.

ex : bash run.sh ./data/context.json ./data/private.json ./output/predictions_private.json
```shell
bash run.sh "${1}" "${2}" "${3}"
```

## Replenishment
* if you didn't change any repository of file, you can directly use the example code to reproduce prediction files.
* By my evaluation process, I produce predictions of the public.json, private.json and put them in the prediction file.


