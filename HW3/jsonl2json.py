import jsonlines
import json
import argparse


parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
parser.add_argument(
    "--jsonl_path",
    type=str,
    default="./data/public.jsonl",
    help="The name of the dataset to use (via the datasets library).",

)
parser.add_argument(
    "--json_path",
    type=str,
    default="./data/public.json",
    help="The name of the dataset to use (via the datasets library).",

)
args = parser.parse_args()
# write_path='./data/public.json'
# read_path='./data/public.jsonl'
with jsonlines.open(args.json_path, "w") as wfd:
    with open(args.jsonl_path, "r", encoding='utf-8') as rfd:
        for data in rfd:
            data = json.loads(data)#注意，这里json文件格式不同，写法也不同，具体看文件,，注意区别json.load()与json.loads()
            wfd.write(data)
print('done !')
