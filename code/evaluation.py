from openai import OpenAI
import os
import jsonlines
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm.rich import tqdm
import time


class Args:
    def parseargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', type=str, default="./results/model_answer.jsonl")
        parser.add_argument('--output_path', type=str, default="./results/model_answer_evaled.jsonl")

        parser.add_argument('--model_name', type=str, default="gpt-4o")
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


eval_prompt = """
You are an expert in diagnosing challenging cases. You will receive a student's answer containing 5 differential diagnoses, as well as the reference diagnosis. You need to score each diagnosis from the student's answer according to the following rules:

2 = The student’s diagnosis exactly matches the reference diagnosis; 
1 = The student’s diagnosis is a broad category that includes the reference diagnosis; 
0 = The student's diagnosis does not meet the criteria for a score of 1 or 2.

Here is the student’s answer: 
%s

Here is the reference diagnosis: 
%s

Output Format: Output the scores in the following format. 
1. Disease 1 Name: \\boxed{The Score of Disease 1};
2. Disease 2 name: \\boxed{The Score of Disease 2};
...
"""


def get_gpt_result_with_retry(item):

    response = client.chat.completions.create(
        model=args.model_name, 
        messages=[
            {"role": "user", "content": eval_prompt % (item["LLM Response"], item["Final Diagnosis"])},
        ]
    )
    text = response.choices[0].message.content
    return_text ={"id": item["id"], "Final Diagnosis": item["Final Diagnosis"], "LLM Response": item["LLM Response"], "response": text}
    with jsonlines.open(args.output_path, mode='a') as writer:
        writer.write(return_text)
    return None



if __name__ == "__main__":

    try:
        if os.path.exists(args.output_path):
            processed_data  = [line['id'] for line in jsonlines.open(args.output_path, mode='r')]
            input_data = [item for item in jsonlines.open(args.input_path, mode='r') if item['id'] not in processed_data]
        else:
            processed_data = []
            input_data = [line for line in jsonlines.open(args.input_path, mode='r')]

        with ThreadPoolExecutor(max_workers=args.folk_nums) as executor:
            list(tqdm(executor.map(get_gpt_result_with_retry, input_data), total=len(input_data)))

    except Exception as e:
        print(e)
        pass