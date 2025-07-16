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
        parser.add_argument('--metric_path', type=str, default="./results/model_mcq_answer.jsonl")

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

    results = []
    for obj in data:

        try:
            answer = re.findall(r"\\boxed{(.*?)}", obj['LLM Response'])[0].strip().lower()
            results.append(1 if answer==obj['Right Option'].lower else 0)

        except Exception as e:
            print(e)
            print(obj)
            exit()
    

    total = len(results)
    correct = sum(results)
    accuracy = correct / total
    error = total - correct
    error_rate = error / total


    # 构造 DataFrame
    df = pd.DataFrame({
        "Model": [f"{model}"],
        "Total": [total],
        "Correct": [correct],
        "Accuracy": [accuracy],
        "Error": [error],
        "Error Rate": [error_rate]
    })

    print(df)



if __name__ == "__main__":

    metric(model=args.model_name, path=args.metric_path)
    
    