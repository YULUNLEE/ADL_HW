import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
import model
from torch.autograd import Variable
from tqdm import trange
import torch.utils.data as Data
import pandas as pd
from dataset import SeqClsDataset
from utils import Vocab
import numpy as np
from tqdm import tqdm
import sys

#
# test_path = sys.argv[1]
# output_csv_path = sys.argv[2]


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
BATCH_SIZE=64
EPOCH=180
LR=1e-3
print('GPU: ', torch.cuda.is_available())
# def main(args):
def main(test_dir, best_model_dir):
    word_dict = {}
    sentence_list = []
    single_sentence_list = []
    total_word_len = 0
    avr_word_len = 0
    label_dict={}
    label_data=[]
    test_id=[]
    single_test_list=[]
    test_list=[]
    test_label=[]
    single_dev_list=[]
    dev_list = []
    dev_label=[]
    pre_dev_label=[]
    # with open(args.cache_dir / "vocab.pkl", "rb") as f:
    #     vocab: Vocab = pickle.load(f)
    #
    # intent_idx_path = args.cache_dir / "intent2idx.json"
    # intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    # print(intent2idx)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    # datasets: Dict[str, SeqClsDataset] = {
    #     split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
    #     for split, split_data in data.items()
    # }
    # print(SeqClsDataset(datasets, vocab, intent2idx, args.max_len).collate_fn(datasets, 4))

    #x_data
    for i in range(len(data['train'][:])):
        total_word_len+=len(data['train'][i]['text'].split(' '))
        # print('word len:',len(datasets['train'][i]['text'].split(' ')))
    avr_word_len=int(total_word_len/(i+1)+6)
    print('average_len=', avr_word_len)

    if '<ED>' not in word_dict:
        word_dict['<ED>'] = 0
    for i in range(len(data['train'][:])):
        for j in range(avr_word_len):
            try:
                single_sentence_list.append(data['train'][i]['text'].split(' ')[j])
                if data['train'][i]['text'].split(' ')[j] not in word_dict:
                # print(datasets['train'][i]['text'].split(' ')[j])
                    word_dict[data['train'][i]['text'].split(' ')[j]]=0
            except:
                single_sentence_list.append('<ED>')
                if '<ED>' not in word_dict:
                    word_dict['<ED>'] = 0
                # print('<ED>')
        sentence_list.append(single_sentence_list)
        single_sentence_list=[]
    # print(sentence_list)
    # print(word_dict)

    #製作自己的glove
    # 將 glove.840B.300d.txt 檔案轉成 dict，key:單字, value:詞向量
    #
    # embeddings_index = {}
    # f = open(r'F:\project\pycharm\pytorch\ADL_HW\HW1\ADL21-HW1-main\glove.840B.300d.txt', encoding='utf8')
    # for line in tqdm(f):
    #     values = line.split()
    #     if len(values) != 301:
    #         continue
    #     # print(i, values)
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float')
    #     embeddings_index[word] = coefs
    #
    # f.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    # # 顯示the單字的詞向量
    # # print(len(embeddings_index["the"]))
    # print(embeddings_index["the"])
    # for i in word_dict.keys():
    #     if i == '<ED>':
    #         word_dict[i] = np.zeros(300, dtype='float').tolist()
    #     else:
    #         try:
    #             word_dict[i]=embeddings_index[i].tolist()
    #         except:
    #             word_dict[i]=np.zeros(300, dtype='float').tolist()
    # print(word_dict.items())
    # with open('my_dict_14.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(word_dict, outfile)

    #load 做好的 glove
    with open('my_dict_14.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
        for i in range(len(sentence_list)):
            for j in range(avr_word_len):
                sentence_list[i][j]=output[sentence_list[i][j]]
        sentence_array=np.array(sentence_list)
        sentence_tensor=torch.FloatTensor(sentence_array)
        print(sentence_tensor.shape)


    # y_data
    num=0
    for i in range(len(data['train'][:])):
        if data['train'][i]['intent'] not in label_dict:
            label_dict[data['train'][i]['intent']]=num
            label_data.append(num)
            num+=1
        else:
            label_data.append(label_dict[data['train'][i]['intent']])
    # print(list(label_dict.keys())[list(label_dict.values()).index(0)])
    # print(label_dict)
    # print(label_data)
    #list-->numpy-->tensor
    label_data=np.array(label_data)
    label_tensor=torch.LongTensor(label_data).view(-1, 1)
    one_hot_label = torch.zeros(len(label_data), len(label_dict)).scatter_(1, label_tensor, 1)
    # print(label_tensor)
    # print(one_hot_label)
    # TODO: init model and move model to target device(cpu / gpu)
    lstm = model.RNN().cuda()
    print(lstm)

    # load model
    lstm = torch.load(best_model_dir)

    # eval
    dev_acc = 0
    print_list=[i for i in range(0, 3000+BATCH_SIZE, BATCH_SIZE)]
    print_test_list = [i for i in range(0, 4500 + BATCH_SIZE, BATCH_SIZE)]
    with open('./data/intent/eval.json', 'r', encoding='utf-8') as f:
        dev_content=json.load(f)
        for i in range(len(dev_content)):
            dev_label.append(dev_content[i]['intent'])
            for j in range(avr_word_len):
                try:
                    single_dev_list.append(dev_content[i]['text'].split()[j])
                except:
                    single_dev_list.append('<ED>')
            dev_list.append(single_dev_list)
            single_dev_list=[]
    # str --> vector
    with open('my_dict_14.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
        for i in range(len(dev_list)):
            for j in range(avr_word_len):
                try:
                    dev_list[i][j]=output[dev_list[i][j]]
                except:
                    dev_list[i][j]=np.zeros(300, dtype='float32')
        dev_list=np.array(dev_list)
        dev_tensor=torch.FloatTensor(dev_list)
    print('dev shape:', dev_tensor.shape)

    for i, j in zip(print_list[0:-1], print_list[1:]):
        dev_out=lstm(dev_tensor[i:j].to('cuda'))
        dev_pred_y = torch.max(dev_out, 1)[1].data.cpu().numpy().tolist()
        for i in dev_pred_y:
            pre_dev_label.append(list(label_dict.keys())[list(label_dict.values()).index(i)])
    # print(dev_label)
    # print(pre_dev_label)
    #dev accuracy
    for i, j in enumerate(dev_label):
        if j == pre_dev_label[i]:
            dev_acc+=1
    dev_acc=dev_acc/len(dev_label)
    print('evaluation accuracy:', dev_acc)


    # TODO: Inference on test set
    with open(test_dir, 'r', encoding='utf-8') as f:
        test_content=json.load(f)
        for i in range(len(test_content)):
            test_id.append(test_content[i]['id'])
            for j in range(avr_word_len):

                try:
                    single_test_list.append(test_content[i]['text'].split()[j])
                except:
                    single_test_list.append('<ED>')
            test_list.append(single_test_list)
            single_test_list=[]

    # str --> vector
    with open('my_dict_14.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
        for i in range(len(test_list)):
            for j in range(avr_word_len):
                try:
                    test_list[i][j]=output[test_list[i][j]]
                except:
                    test_list[i][j]=np.zeros(300, dtype='float32')
        test_list=np.array(test_list)
        test_tensor=torch.FloatTensor(test_list)
    print('test shape:', test_tensor.shape)

    for i, j in zip(print_test_list[0:-1], print_test_list[1:]):
        test_out=lstm(test_tensor[i:j].to('cuda'))
        test_pred_y = torch.max(test_out, 1)[1].data.cpu().numpy().tolist()
        for i in test_pred_y:
            test_label.append(list(label_dict.keys())[list(label_dict.values()).index(i)])
    #
    # test_out=lstm(test_tensor.to('cuda'))
    # test_pred_y = torch.max(test_out, 1)[1].data.cpu().numpy().tolist()
    # for i in test_pred_y:
    #     test_label.append(list(label_dict.keys())[list(label_dict.values()).index(i)])
    # print(test_label)
    return test_label, test_id





def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Directory to the testdata.",
        default = "./data/intent/test.json"

    )

    parser.add_argument(
        "--kaggle_dir",
        type=Path,
        help="Directory to the csv_output.",
        default="kaggle_submission_intent.csv"

    )

    parser.add_argument(
        "--best_model_dir",
        type=Path,
        help="Directory to the best model.",
        default="./intent_best91.33.pkl"

    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("test_data_dir:", args.test_dir)
    print("kaggle_output_dir:", args.kaggle_dir)
    print("best_model_dir:", args.best_model_dir)
    # main(args)
    output_ans, output_id = main(args.test_dir, args.best_model_dir)
    # print('id:', output_id)
    # print('ans:', output_ans)
    dict = {"id": output_id, "intent": output_ans}
    df = pd.DataFrame(dict, columns=["id", "intent"])
    df.to_csv(args.kaggle_dir, index=0)
    print("kaggle_submission done !")


