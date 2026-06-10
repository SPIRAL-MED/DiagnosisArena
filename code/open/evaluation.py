from openai import OpenAI
import os
import jsonlines
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.rich import tqdm
import time


class Args:
    def parseargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', type=str,
                            help="Path to the inference output .jsonl file to be evaluated.")

        parser.add_argument('--judge_model', type=str,
                            help="Model name of the judge LLM used for scoring. Appended to input filename as _{judge_model}_evaled.jsonl.")
        parser.add_argument("--api_key", type=str, default=None,
                            help="API key for the judge LLM provider. Defaults to the OPENAI_API_KEY environment variable if not set.")
        parser.add_argument("--base_url", type=str, default=None,
                            help="Base URL of the judge LLM API endpoint. Use this to point to a custom or self-hosted endpoint.")
        parser.add_argument('--folk_nums', type=int, default=16,
                            help="Number of parallel threads for evaluation. Tune according to your API rate limits.")

        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

        self.output_path = self.input_path.removesuffix(".jsonl") + f"_{self.judge_model}_evaled.jsonl"

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
        model=args.judge_model,
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
            futures = {executor.submit(get_gpt_result_with_retry, item): item for item in input_data}
            with tqdm(total=len(futures), desc=f"Evaluation with [{args.judge_model}]") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    except Exception as e:
        print(e)
        pass