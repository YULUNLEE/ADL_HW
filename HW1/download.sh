wget https://www.dropbox.com/s/ozeiq17yxd18qxn/my_dict_14.json?dl=1 -O my_dict_14.json
wget https://www.dropbox.com/s/dglg0n44c8ufo85/my_token_dict_35.json?dl=1 -O my_token_dict_35.json
wget https://www.dropbox.com/s/t7jxjn273n7q3dt/tagging_best82.252.pkl?dl=1 -O tagging_best82.252.pkl
wget https://www.dropbox.com/s/2mv0vd7voa35vlb/intent_best91.33.pkl?dl=1 -O intent_best91.33.pkl
mkdir data cache
cd data
mkdir intent slot
cd intent
wget https://www.dropbox.com/s/pylz2s5zh9y4lcw/eval.json?dl=1 -O eval.json
wget https://www.dropbox.com/s/esq123oloopjcu0/test.json?dl=1 -O test.json
wget https://www.dropbox.com/s/8c9tw84q6rpm3gn/train.json?dl=1 -O train.json
cd ..
cd slot
wget https://www.dropbox.com/s/92ivppmy4ftx1sd/eval.json?dl=1 -O eval.json
wget https://www.dropbox.com/s/gk5nc65558cb2m7/test.json?dl=1 -O test.json
wget https://www.dropbox.com/s/simiv8qkzka3fot/train.json?dl=1 -O train.json

cd ../..
cd cache
mkdir intent slot
cd slot
wget https://www.dropbox.com/s/ty1jyzabtf1krxt/my_tag2idx.json?dl=1 -O my_tag2idx.json
