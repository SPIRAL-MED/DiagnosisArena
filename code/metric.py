import json
import re
import argparse
import numpy as np
import copy
from rich import print
import jsonlines
import pandas as pd


class Args:
    def parseargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default="gpt-4o")
        parser.add_argument('--metric_path', type=str, default="./results/model_answer_evaled.jsonl")

        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

    def __init__(self) -> None:
        self.parseargs()
args = Args()


def metric(model, path):

    with open(path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
        data = sorted(data, key=lambda x: x['id'])

    top1 = []
    top5 = []
    for obj in data:

        try:
            scores = re.findall(r"\\boxed{(.*?)}", obj['response'])
            scores = [int(score) for score in scores]
            top1.append(0 if scores[0]<=1 else 1)
            top5.append(0 if max(scores[:5]) <=1 else 1)

        except Exception as e:
            print(e)
            print(obj)
            exit()
    

    total_top1 = len(top1)
    correct_top1 = sum(top1)
    accuracy_top1 = correct_top1 / total_top1
    error_top1 = total_top1 - correct_top1
    error_rate_top1 = error_top1 / total_top1

    total_top5 = len(top5)
    correct_top5 = sum(top5)
    accuracy_top5 = correct_top5 / total_top5
    error_top5 = total_top5 - correct_top5
    error_rate_top5 = error_top5 / total_top5

    # 构造 DataFrame
    df = pd.DataFrame({
        "Metric": [f"{model} Top 1", f"{model} Top 5"],
        "Total": [total_top1, total_top5],
        "Correct": [correct_top1, correct_top5],
        "Accuracy": [accuracy_top1, accuracy_top5],
        "Error": [error_top1, error_top5],
        "Error Rate": [error_rate_top1, error_rate_top5]
    })

    print(df)



if __name__ == "__main__":

    metric(model=args.model_name, path=args.metric_path)
    
    