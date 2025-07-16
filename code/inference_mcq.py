import json
import os
import argparse
import jsonlines

from openai import OpenAI
from rich import print
from tqdm.rich import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset



class Args:
    def parseargs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--hf_data_path', type = str, default="shzyk/DiagnosisArena")
        parser.add_argument('--output_path', type=str, default="./results/model_answer.jsonl")

        parser.add_argument("--model_name", type=str, default="gpt-4o", help="The name of the LLM model to use for inference.")
        parser.add_argument("--api_key", type=str, default=None)
        parser.add_argument("--base_url", type=str, default=None)

        parser.add_argument('--folk_nums', type=int, default=16, help="The number of threads to use for inference. It depends on the LLM api you use.")

        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

    def __init__(self) -> None:
        self.parseargs()
args = Args()



client = OpenAI(
    api_key=args.api_key,
    base_url=args.base_url
)


inference_prompt = \
"""
According to the provided medical case and select the most appropriate diagnosis from the following four options. Put your final answer within \\boxed{}.

Here is the medical case: 
Case Information:
%s
Physical Examination:
%s
Diagnostic Tests:
%s

Here are the four options: 
%s

Put your final answer letter within \\boxed{}.
Final answer: \\boxed{Correct Option Letter}
"""


def llm_folk(item: dict):

    response = client.chat.completions.create(
        model=args.model_name, 
        messages=[
            {"role": "user", "content": inference_prompt % (item["Case Information"], item["Physical Examination"], item["Diagnostic Tests"], item["Options"])},
        ]
    )
    text = response.choices[0].message.content
    with jsonlines.open(args.output_path, mode='a') as writer:
        writer.write({"id": item["id"], "Right Option": item["Right Option"], "LLM Response": text})  


if __name__ == "__main__":


    try:
        input_datas=load_dataset(args.hf_data_path, split="test")
        # input_datas = input_datas.select(range(10))

        if os.path.exists(args.output_path):
            with jsonlines.open(args.output_path, mode='r') as reader:
                generated_datas = [obj for obj in reader]
            generated_ids = set([g['id'] for g in generated_datas])

            rest_datas = []
            for d in input_datas:
                if d['id'] not in generated_ids:
                    rest_datas.append(d)
        else:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            rest_datas = input_datas

        with ThreadPoolExecutor(max_workers=args.folk_nums) as executor:
            list(tqdm(executor.map(llm_folk, rest_datas), total=len(rest_datas)))

    except Exception as e:

        print(e)
        pass

