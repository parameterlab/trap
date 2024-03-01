"""
Generate CSV of empirical distribution of answers from API models
"""
import argparse
import os
import random
import pandas as pd

import json
import re
from tqdm import tqdm

from utils import save_csv, load_system_prompts
from openai import OpenAI



def openai_sample_once(model, user_prompt, system_prompt, temperature=1.0, top_p=1.0, max_tokens=64):
    if system_prompt is None or user_prompt is None:
        raise ValueError('prompts cannot be None')
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        tool_choice=None,  # do not call fn, generate output
    )
    completion = completion.choices[0].message.content
    return completion


def generate_n_times(api, desired_size, user_prompt, model_name, system_prompt, temperature, top_p, n=10, seed=None, verbose=0):
    n_reject, answers = 0, []
    with tqdm(total=n) as pbar:
        while len(answers) < n:
            if api == 'openai':
                completion = openai_sample_once(model=model_name, user_prompt=user_prompt, system_prompt=system_prompt,
                                   temperature=temperature, top_p=top_p)
            else:
                raise NotImplementedError('API not supported')

            # extract the fist N digits in the answer
            num_gen_search = re.search(r'(\d{'+str(desired_size)+'})', completion)  # the number is eihter at the start of the string or after a space
            num_gen = num_gen_search.group(1) if num_gen_search else None
            if num_gen is not None:
                answers.append(num_gen)
                pbar.update(1)
            # no answer or answer not valid
            if num_gen is None:
                n_reject += 1
                if verbose >= 2:
                    print(f"Number extraction failed for: {completion}\n")
            if verbose >= 5:
                print(f"Num generated: {num_gen} (completion: {completion})")
    if verbose >= 1:
        print(f'* Nb valid answers: {len(answers)}. Invalid answers: {n_reject}/{n} = {100 * n_reject / n:.2f}%')
    return n_reject, n, answers


def main():
    parser = argparse.ArgumentParser(description="Sample answers from a model.")
    parser.add_argument("--api", default='openai', help="API name")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--export-base-folder", default='.', help="base directory to export csv")
    parser.add_argument("--export-sub-folder", default=None, help="create a subdirectory to export csv")
    parser.add_argument("--n-gen", default=1000, type=int, help="Number of answers to sample.")
    parser.add_argument("--n-digits", default=4, type=int, help="Number of digits of the answer.")
    parser.add_argument("--system-prompt", default=None, help="Name of the system prompt to use. Does NOT support 'all'. Default (None), load the default model system prompt.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose (at =2 print completions that failed to extract number).")
    args = parser.parse_args()

    desired_size = args.n_digits
    target = f"Sure, here is a random string of {desired_size} digits: "
    user_prompt = f"Write a random string composed of {desired_size} digits. Your reply should only contain the random string."

    system_prompts_dict = load_system_prompts(name=args.system_prompt, model_name=args.api)
    scenario = args.system_prompt
    system_prompt = system_prompts_dict['original' if scenario is None else scenario]
    if scenario != 'original':
        print(f'Scenario: {scenario}')

    n_reject, n, answers_list = generate_n_times(api=args.api, desired_size=desired_size, user_prompt=user_prompt,
                                                 model_name=args.model_name, system_prompt=system_prompt,
                                                 temperature=args.temperature, top_p=args.top_p, n=args.n_gen,
                                                verbose=args.verbose)
    if args.temperature != 1.0:
        print(f'Temperature config: {args.temperature}')
    if args.top_p != 1.0:
        print(f'Top_p config: {args.top_p}')

    # individual answers
    df_answers = pd.DataFrame({'answer': answers_list})

    # export individual answers
    path = os.path.join(args.export_base_folder, 'results/baseline/answers_nosuffix/', args.api+'_'+args.model_name)
    if args.export_sub_folder:
        path = os.path.join(path, args.export_sub_folder)
    filename = f"answers_samples_{args.n_digits}digits{'_system_prompt_'+args.system_prompt if args.system_prompt else ''}{'_temperature_'+str(args.temperature) if args.temperature else ''}{'_top_p_'+str(args.top_p) if args.top_p else ''}.csv"
    path_answers = os.path.join(path, filename)
    save_csv(df_answers, path_answers)
    print(f'Shape: {df_answers.shape}')
    print(f'Top answers:\n{ df_answers["answer"].value_counts()}')


if __name__ == "__main__":
    main()
