wget https://www.dropbox.com/s/9h0zas78ns5hcv9/config.json?dl=1 -O config.json
wget https://www.dropbox.com/s/oodqvc0jrynntkh/sample_train.json?dl=1 -O sample_train.json
mkdir data

cd data
wget https://www.dropbox.com/s/euu14jdnxbdxep0/context.json?dl=1 -O context.json
wget https://www.dropbox.com/s/zagt948l6mfzhod/train.json?dl=1 -O train.json
wget https://www.dropbox.com/s/n6h57ei7pfimfbg/public.json?dl=1 -O public.json
wget https://www.dropbox.com/s/ubuk2vkq1550ndq/private.json?dl=1 -O private.json
wget https://www.dropbox.com/s/axeuan3zgwkm4u6/squad_train.json?dl=1 -O squad_train.json
wget https://www.dropbox.com/s/unvuas0ltcgaw50/vocab.txt?dl=1 -O vocab.txt
cd ..

mkdir ft_dir
cd ft_dir
wget https://www.dropbox.com/s/35r07ooawhp6pnm/ft_model.bin?dl=1 -O ft_model.bin
cd ..

mkdir output

mkdir processed

mkdir prediction
cd prediction
wget https://www.dropbox.com/s/4omnm2pjud4y3rm/pred_public.json?dl=1 -O public.json
wget https://www.dropbox.com/s/ysjd3oycpoff86w/pred_private.json?dl=1 -O private.json

cd..