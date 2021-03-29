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
from tqdm import tqdm, trange
from seqeval_master.seqeval.scheme import IOB2
from seqeval_master.seqeval.metrics import v1
from seqeval_master.seqeval.metrics import sequence_labeling

TRAIN = "train"
DEV = "eval"
TEST='test'
# SPLITS = [TRAIN, DEV, TEST]
SPLITS = [TRAIN, DEV]
BATCH_SIZE=32
EPOCH=150
LR=1e-3
print('GPU: ', torch.cuda.is_available())
# def main(args):
def main(test_dir):
    # with open(args.cache_dir / "vocab.pkl", "rb") as f:
    #     vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "my_tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    # print(intent2idx)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    test_paths = test_dir
    test_data = json.loads(test_paths.read_text())
    # print(test_paths)
    # print('1', test_data)
    # datasets: Dict[str, SeqClsDataset] = {
    #     split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
    #     for split, split_data in data.items()
    # }

    #x preprocessing

    train_token_list=[]
    word_dict={}
    avg_len=0
    single_token_list=[]
    all_token_list=[]
    single_tags_list=[]
    all_tags_list = []
    for i in range(len(data['train'])):
        tokens=data['train'][i]['tokens']
        train_token_list.append(tokens)
    # print(train_token_list)


    for i in range(len(train_token_list)):
        if len(data['train'][i]['tokens']) > avg_len:
            avg_len=len(data['train'][i]['tokens'])
    print('max_len= ', avg_len)

    for i in range(len(train_token_list)):
        for j in range(avg_len):
            try:
                single_token_list.append(train_token_list[i][j])
            except:
                single_token_list.append('<ED>')
        all_token_list.append(single_token_list)
        single_token_list=[]
    # print(all_token_list)

    #做 word_dict
    for i in range(len(all_token_list)):
        for j in range(avg_len):
            if all_token_list[i][j] not in word_dict:
                word_dict[all_token_list[i][j]]=0
    # print(word_dict)

    # 製作自己的glove
    # 將 glove.840B.300d.txt 檔案轉成 dict，key:單字, value:詞向量
    # embeddings_index = {}
    # f = open(r'F:\project\pycharm\pytorch\ADL_HW\HW1\ADL21-HW1-main\glove.840B.300d.txt', encoding='utf8')
    # for line in tqdm(f):
    #     values = line.split(" ")
    #     values[-1].strip('\n')
    #     if len(values) != 301:
    #         print(values)
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
    #             word_dict[i] = embeddings_index[i].tolist()
    #         except:
    #             word_dict[i] = np.zeros(300, dtype='float').tolist()
    # print(word_dict.items())
    # with open('my_token_dict_35.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(word_dict, outfile)

    # load 做好的 glove
    with open('my_token_dict_35.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
        for i in range(len(all_token_list)):
            for j in range(avg_len):
                all_token_list[i][j] = output[all_token_list[i][j]]
        torken_array = np.array(all_token_list)
        torken_tensor = torch.FloatTensor(torken_array)
        # print(torken_tensor)
        print("token shape : ", torken_tensor.shape)

    #y data
    #label : str
    for i in range(len(data['train'])):
        for j in range(avg_len):
            try:
                single_tags_list.append(data['train'][i]['tags'][j])
            except:
                single_tags_list.append("<ED>")
        all_tags_list.append(single_tags_list)
        single_tags_list = []
    # print(all_tags_list)

    #lable : str --> int
    for i in range(len(all_tags_list)):
        for j in range(avg_len):
            all_tags_list[i][j]=intent2idx[all_tags_list[i][j]]
    # print(all_tags_list)
    all_tags_numpy=np.array(all_tags_list)
    all_tags_tensor=torch.LongTensor(all_tags_numpy)
    print(all_tags_tensor)
    print("tags shape : ", all_tags_tensor.shape)


    #do dataset / dataloader
    dataset = Data.TensorDataset(torken_tensor, all_tags_tensor)
    # loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)

    #model
    # gru = model.SlotRNN(BATCH_SIZE, 300, 512, 10).cuda()
    lstm = model.SlotRNN().cuda()
    # optimizer = torch.optim.Adam(gru.parameters(), lr=LR)
    optimizer = torch.optim.AdamW(lstm.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,amsgrad=False)
    criterion = torch.nn.CrossEntropyLoss()
    print(lstm)


    #training

    for epoch in range(EPOCH):
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
        # h_n = Variable(torch.randn(2 * 2, BATCH_SIZE, 1024).to('cuda')) # 同样考虑向前层和向后层
        # h_c = Variable(torch.randn(2 * 2, BATCH_SIZE, 1024).to('cuda')) # (num_layers * num_directions, batch_size, hidden_size)
        print_loss = 0
        train_pred_label = []
        total_acc = 0
        loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
        for step, (b_x, b_y) in enumerate(loader):
            # zero_grad
            optimizer.zero_grad()
            b_x = b_x.view(-1, avg_len, 300).to('cuda')  # reshape x to (batch, time_step, input_size)
            pred = lstm(b_x)  # rnn output
            train_pred_label.append(var2np(pred))
            # print('1', pred.shape)
            # print('2', b_y.shape)
            # compute loss
            loss = criterion(pred, b_y.view(-1).to('cuda'))
            print_loss += loss.item()
            #accuracy
            flag = 0

            for i in range(len(train_pred_label[0])):  #len(train_pred_label[0])=35*Batch_size
                # print(b_y.view(-1).data.cpu().numpy())
                if train_pred_label[step][i]==b_y.view(-1)[i].data.cpu().numpy():
                    if (i + 1) % 35 == 0 and flag == 0:
                        # acc+=1
                        total_acc += 1
                else:
                    flag=1
                    if (i + 1) % 35 == 0:
                        flag=0
                    # break
            # total_acc+=acc
            # backward

            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # print(len(loader)) #Total/BATCH_SIZE
        train_acc = total_acc / len(loader) * 100/BATCH_SIZE
        print (f'epoch: {epoch+1} / {EPOCH} | loss: {print_loss/len(loader)} | total_acc: {total_acc} | train_accuracy: {train_acc}% ')
        # print(train_pred_label[-1])
        # print(b_y.view(-1).data.cpu().numpy())


    # save model
    torch.save(lstm, 'tag_model.pkl')

    # #eval
    # # dev_acc = 0
    # # print_list = [i for i in range(0, 1000 + BATCH_SIZE, BATCH_SIZE)]
    # # print_test_list = [i for i in range(0, 4500 + BATCH_SIZE, BATCH_SIZE)]
    dev_token_list=[]
    single_dev_token_list=[]
    all_dev_token_list=[]
    pre_dev_token_label=[]
    pre_dev_token_label_2D=[]
    pre_dev_token_label_clear=[]
    pre_dev_token_label_2D_clear=[]
    ground_truth_dev_label=[]
    total_eval_correct=0
    # dev_pred_y=[]
    for i in range(len(data['eval'])):
        tokens = data['eval'][i]['tokens']
        dev_token_list.append(tokens)
    # print(dev_token_list)
    for i in range(len(dev_token_list)):
        for j in range(avg_len):
            try:
                single_dev_token_list.append(dev_token_list[i][j])
            except:
                single_dev_token_list.append('<ED>')
        all_dev_token_list.append(single_dev_token_list)
        single_dev_token_list = []

    # load 做好的 glove
    with open('my_token_dict_35.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
        for i in range(len(all_dev_token_list)):
            for j in range(avg_len):
                try:
                    all_dev_token_list[i][j] = output[all_dev_token_list[i][j]]
                except:
                    all_dev_token_list[i][j] = np.zeros(300, dtype='float32')
        dev_torken_array = np.array(all_dev_token_list)
        dev_torken_tensor = torch.FloatTensor(dev_torken_array)
        # print(torken_tensor)
        print("dev token shape : ", dev_torken_tensor.shape)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,profile_memory=False) as prof:
        dev_out = lstm(dev_torken_tensor.to('cuda'))
    print(prof.table())
    # print(len(var2np(dev_out)))

    for i, j in enumerate(var2np(dev_out)):
        pre_dev_token_label.append(list(intent2idx.keys())[list(intent2idx.values()).index(j)])
        if (i+1)%35==0:
            pre_dev_token_label_2D.append(pre_dev_token_label)
            pre_dev_token_label=[]
    # print(pre_dev_token_label_2D)

    #刪掉 '<ED>'
    for i in range(len(pre_dev_token_label_2D)):
        for j in range(avg_len):
            if pre_dev_token_label_2D[i][j]!='<ED>':
                pre_dev_token_label_clear.append(pre_dev_token_label_2D[i][j])
        pre_dev_token_label_2D_clear.append(pre_dev_token_label_clear)
        pre_dev_token_label_clear=[]
    # print('1',pre_dev_token_label_2D_clear)

    #load ground truth eval tags
    for i in range(len(data['eval'])):
        ground_truth_dev_label.append(data['eval'][i]['tags'])
    # print('2', ground_truth_dev_label)
    # print('3',ground_truth_dev_label[1]==pre_dev_token_label_2D_clear[1])

    # 字數處理成相同

    for i in range(len(ground_truth_dev_label)):
        while True:
            if len(ground_truth_dev_label[i]) > len(pre_dev_token_label_2D_clear[i]):
                pre_dev_token_label_2D_clear[i].append('O')
            elif len(ground_truth_dev_label[i]) < len(pre_dev_token_label_2D_clear[i]):
                pre_dev_token_label_2D_clear[i].pop()
            else:
                break

    #eval_accuracy

    for i in range(len(data['eval'])):
        if ground_truth_dev_label[i]==pre_dev_token_label_2D_clear[i]:
            total_eval_correct+=1
    eval_acc=total_eval_correct/len(data['eval']) *100
    print(f"eval joint accuracy : {eval_acc}%")


    # test
    test_token_list=[]
    test_avg_len=0
    single_test_token_list=[]
    all_test_token_list=[]
    pre_test_token_label=[]
    pre_test_token_label_2D=[]
    pre_test_token_label_clear=[]
    pre_test_token_label_2D_clear=[]
    test_id_list=[]
    for i in range(len(test_data)):
        id=test_data[i]['id']
        tokens = test_data[i]['tokens']
        test_id_list.append(id)
        test_token_list.append(tokens)
    # print(test_token_list)

    # max_test_len=35
    # for i in range(len(test_token_list)):
    #     if len(data['test'][i]['tokens']) > test_avg_len:
    #         test_avg_len=len(data['test'][i]['tokens'])
    # print('max_len= ', test_avg_len)

    for i in range(len(test_token_list)):
        for j in range(avg_len):
            try:
                single_test_token_list.append(test_token_list[i][j])
            except:
                single_test_token_list.append('<ED>')
        all_test_token_list.append(single_test_token_list)
        single_test_token_list = []
    # print(all_test_token_list)

    # load 做好的 glove
    with open('my_token_dict_35.json', 'r', encoding='utf-8') as f:
        output = json.load(f)
        for i in range(len(all_test_token_list)):
            for j in range(avg_len):
                try:
                    all_test_token_list[i][j] = output[all_test_token_list[i][j]]
                except:
                    all_test_token_list[i][j] = np.zeros(300, dtype='float32')
        test_torken_array = np.array(all_test_token_list)
        test_torken_tensor = torch.FloatTensor(test_torken_array)
        # print(torken_tensor)
        print("test token shape : ", test_torken_tensor.shape)

    test_out = lstm(test_torken_tensor.to('cuda'))
    # print(len(var2np(test_out)))

    for i, j in enumerate(var2np(test_out)):
        pre_test_token_label.append(list(intent2idx.keys())[list(intent2idx.values()).index(j)])
        if (i+1)%35==0:
            pre_test_token_label_2D.append(pre_test_token_label)
            pre_test_token_label=[]
    # print(pre_test_token_label_2D)

    #刪掉 '<ED>'
    for i in range(len(pre_test_token_label_2D)):
        for j in range(avg_len):
            if pre_test_token_label_2D[i][j]!='<ED>':
                pre_test_token_label_clear.append(pre_test_token_label_2D[i][j])
        pre_test_token_label_2D_clear.append(pre_test_token_label_clear)
        pre_test_token_label_clear=[]

    # #把字數處理成相同
    for i in range(len(test_token_list)):
        while True:
            if len(test_token_list[i]) > len(pre_test_token_label_2D_clear[i]):
                pre_test_token_label_2D_clear[i].append('O')
            elif len(test_token_list[i]) < len(pre_test_token_label_2D_clear[i]):
                pre_test_token_label_2D_clear[i].pop()
            else:
                break

    return test_id_list, pre_test_token_label_2D_clear, ground_truth_dev_label, pre_dev_token_label_2D_clear


def var2np(variable):
    return torch.max(variable, 1)[1].data.cpu().squeeze(0).numpy()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Directory to the testdata.",
        default="./data/slot/test.json",

    )

    parser.add_argument(
        "--kaggle_dir",
        type=Path,
        help="Directory to the csv_output.",
        default="kaggle_submission_tags.csv"

    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
    test_output_str = ''
    test_output_modify=[]
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("test_data_dir:", args.test_dir)
    print("kaggle_output_dir:", args.kaggle_dir)
    # main(args)
    test_id, test_output, eval_groundTruth, eval_pred = main(args.test_dir)
    # print('id:', test_id)
    # print('test_output', test_output)
    print(v1.classification_report(eval_groundTruth, eval_pred, scheme=IOB2))
    print(f'Token accuracy: {sequence_labeling.accuracy_score(eval_groundTruth, eval_pred) * 100} %')
    for i in range(len(test_output)):
        for j in range(len(test_output[i])):
            test_output_str += test_output[i][j]
            if j!=len(test_output[i])-1:
                test_output_str += ' '
        test_output_modify.append(test_output_str)
        test_output_str = ''

    dict = {'id': test_id, 'tags': test_output_modify}
    df = pd.DataFrame(dict, columns=["id", "tags"])
    df.to_csv(args.kaggle_dir, index=0)
