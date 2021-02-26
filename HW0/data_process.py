import pandas as pd
import numpy as np



longword_dict={}
def dict_clearValue(word_dict):
    for i in word_dict:
        word_dict[i]=0


def data_processing():
    word_dict={}
    train_data=[]
    df = pd.read_csv('train.csv', encoding='utf-8')
    word=df['text']
    train_label=df['Category']


    for i in word:
        for j in i.split(' '):
            # print(j)
            if j not in word_dict :
                word_dict[j]=1
            else:
                word_dict[j]+=1
    print("未刪減過後長度:", len(word_dict))
    print("train_date 數量:", len(word))
    #刪掉出現少次數的

    for i in word_dict:
        # print(word_dict[i])
        if int(word_dict[i])>50 and len(i)<8:
            longword_dict[i]=word_dict[i]
    print("刪減過後長度:", len(longword_dict))
    print("train_date 數量:", len(word))
    # print(longword_dict)

    #bag of word vector
    for i in word:
        for j in i.split(' '):
            # print(j)
            if j in longword_dict:
                longword_dict[j] += 1
        # print(list(longword_dict.values()))
        train_data.append(list(longword_dict.values()))
        dict_clearValue(longword_dict)
    train_data=np.array(train_data)
    # print(train_data)
    # print(train_data.shape)
    train_label=np.array(train_label)[:, np.newaxis]
    # print(train_label.shape)

    return train_data, train_label, len(longword_dict)



def test_data_peocessing():
    test_data = []
    df = pd.read_csv('dev.csv', encoding='utf-8')
    word = df['text']
    test_label = df['Category']


    # bag of word vector
    for i in word:
        for j in i.split(' '):

            if j in longword_dict:
                longword_dict[j] += 1
        # print(list(longword_dict.values()))
        test_data.append(list(longword_dict.values()))
        dict_clearValue(longword_dict)
    test_data = np.array(test_data)
    test_label = np.array(test_label)[:, np.newaxis]
    print(test_data.shape)
    # print(test_label)
    return test_data, test_label



if __name__ == '__main__':
    data_processing()  # 或是任何你想執行的函式
    test_data_peocessing()





