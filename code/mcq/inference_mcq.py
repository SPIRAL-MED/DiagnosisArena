import json
import os
import argparse
import jsonlines

from openai import OpenAI
from datasets import load_dataset
from rich import print
from tqdm.rich import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class Args:
    def parseargs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--input_path', type=str,
                            help="Input data path. Can be a local .jsonl file path or a HuggingFace dataset repo ID (e.g. 'shzyk/DiagnosisArena').")
        parser.add_argument('--output_root', type=str,
                            help="Root directory for output files. The output file will be saved as {output_root}/{model_name}_answer.jsonl.")

        parser.add_argument("--model_name", type=str,
                            help="Model name used for inference and as part of the output filename.")
        parser.add_argument("--api_key", type=str, default=None,
                            help="API key for the LLM provider. Defaults to the OPENAI_API_KEY environment variable if not set.")
        parser.add_argument("--base_url", type=str, default=None,
                            help="Base URL of the LLM API endpoint. Use this to point to a custom or self-hosted endpoint.")

        parser.add_argument('--folk_nums', type=int, default=16,
                            help="Number of parallel threads for inference. Tune according to your API rate limits.")

        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

        self.output_path = f"{self.output_root}/{self.model_name}_mcq_answer.jsonl"

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
        if os.path.isfile(args.input_path):
            input_datas = [line for line in jsonlines.open(args.input_path, mode='r')]
        else:
            input_datas = list(load_dataset(args.input_path, split="test"))

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
            futures = {executor.submit(llm_folk, item): item for item in rest_datas}
            with tqdm(total=len(futures), desc=f"Inference MCQ [{args.model_name}]") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    except Exception as e:

        print(e)
        pass

