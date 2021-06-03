mkdir best_model
cd best_model
wget https://www.dropbox.com/s/v93p8x9g6fihzqm/config.json?dl=1 -O config.json
wget https://www.dropbox.com/s/vpql7f6t1jup8sk/pytorch_model.bin?dl=1 -O pytorch_model.bin
cd ..
mkdir data
cd data
wget https://www.dropbox.com/s/pnm4ftvq44fcajo/public.jsonl?dl=1 -O public.jsonl
wget https://www.dropbox.com/s/gisvtamrtie6bbm/train.jsonl?dl=1 -O train.jsonl
cd ..

mkdir output
