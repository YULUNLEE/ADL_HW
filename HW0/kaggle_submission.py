import pandas as pd
from ADL_HW.HW0 import MLP_model


id_list=[]
label_list=[]

sample_df = pd.read_csv('sample_submission.csv', encoding='utf-8')
ids=sample_df["Id"]

answer_list=MLP_model.model()

for id in ids:
    id_list.append(id)

for i in answer_list:
    label_list.append(int(i))

print(len(id_list))
print(len(label_list))
dict={"Id":id_list, "Category":label_list}
df = pd.DataFrame(dict, columns = ["Id", "Category"])
df.to_csv("kaggle_sb6.csv", index=0)
print(df)

#
# #data 合併
# id_list=[]
# word_list=[]
# label_list=[]
#
# train_df = pd.read_csv('train.csv', encoding='utf-8')
# dev_df = pd.read_csv('dev.csv', encoding='utf-8')
#
# train_id=train_df["Id"]
# dev_id=dev_df["Id"]
# train_word=train_df["text"]
# dev_word=dev_df["text"]
# train_label=train_df["Category"]
# dev_label=dev_df["Category"]
#
# for i in train_id:
#     id_list.append(i)
# for i in dev_id:
#     id_list.append(i)
#
# for i in train_word:
#     word_list.append(i)
# for i in dev_word:
#     word_list.append(i)
#
# for i in train_label:
#     label_list.append(i)
# for i in dev_label:
#     label_list.append(i)
#
#
#
# dict={"Id":id_list, "text":word_list, "Category":label_list}
# df = pd.DataFrame(dict, columns = ["Id", "text", "Category"])
# df.to_csv("train_concat.csv", index=0)
# print(df)
