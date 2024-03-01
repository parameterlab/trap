import argparse
import os
import random
import json
import numpy as np
import pandas as pd
import re

from prompttools.experiment import OpenAIChatExperiment, AnthropicCompletionExperiment
from anthropic import HUMAN_PROMPT, AI_PROMPT

from utils import create_parent_folder, load_suffixes, save_csv, get_datetime, load_system_prompts

API_NAMES = ['openai', 'anthropic']

def main():
    parser = argparse.ArgumentParser(description="Completion from API LLM from JSON suffixes.")
    parser.add_argument("-p", "--path-suffixes", required=True,  help="Path to the folder with JSON files of suffixes")
    parser.add_argument("-m", "--model-name",  help="Name of the model")
    parser.add_argument("-a", "--api-name", choices=API_NAMES,  help="Type of API")
    parser.add_argument("-f", "--export-csv", default=None,  help="Export to this file")
    parser.add_argument("-n", "--n-gen", default=10, type=int, help="Number of answers to generate for each suffix.")
    #parser.add_argument("-s", "--string-type", choices=['number', 'string'], help="Type of goal string.")
    parser.add_argument("-y", "--system-prompt", default=None, help="Name of the system prompt to use. 'all' tries all the available system prompts. Default (None), load the default model system prompt.")
    parser.add_argument("-g", "--gen-config-override", default=None, help="Override generation config with the provided values. Default (None), load the default model gen config. Each element should be a list. Example: `{'temperature': [0.6, 1.0], 'top_p': [0.99] }`")
    parser.add_argument("-s", "--n-suffixes", default=None, type=int, help="Complete the first n suffixes.")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose (at =2 print completions that failed to extract number).")
    args = parser.parse_args()

    model_suffix = re.search(r'/model_([^/]+)/', args.path_suffixes).group(1) if re.search(r'/model_([^/]+)/', args.path_suffixes) else args.path_suffixes

    if not args.export_csv:
        args.export_csv = os.path.join(args.path_suffixes, 'api', f"retrieval_rate_{args.api_name}{'_system_prompts' if args.system_prompt else ''}.csv")

    system_prompts_dict = load_system_prompts(name=args.system_prompt, model_name=args.api_name)  # we pass api name

    df_suffixes = load_suffixes(args.path_suffixes)
    if args.n_suffixes:
        df_suffixes = df_suffixes.iloc[:args.n_suffixes,:]
    df_suffixes['full_prompt'] = df_suffixes.apply(lambda row: str(row['goals']) + ' ' +  str(row['control']), axis=1).astype(str)
    list_adv_suffix = df_suffixes['control'].to_list()
    list_target_number = df_suffixes['number'].to_list()

    desired_size = len(list_target_number[0])
    #user_prompt = f"Write a random string composed of {desired_size} digits. Your reply should only contain the random string."

    if args.gen_config_override:
        try:
            gen_config_override = json.loads(args.gen_config_override.replace("'", '"'))  # json requires double quotes
        except (ValueError, SyntaxError) as e:
            print("[ERROR] invalid json to override generation config")
            raise e
    else:
        gen_config_override = {}

    for scenario, system_prompt in system_prompts_dict.items():
        if scenario != 'original':
            print(f'*** SCENARIO: {scenario} ***')
        df_all = pd.DataFrame()
        for i in range(args.n_gen):
            #print(f'* Generation #{i+1} *')

            # load api
            if args.api_name == 'openai':
                messages = [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{user_prompt} {suffix}"},
                    ]
                    for suffix, user_prompt in zip(df_suffixes['control'], df_suffixes['goals'])
                ]
                experiment = OpenAIChatExperiment([args.model_name], messages,
                                                  n=[1], max_tokens=[64],
                                                  **gen_config_override
                                                  )
            elif args.api_name == 'anthropic':
                messages = [
                    f"{system_prompt}{HUMAN_PROMPT}{user_prompt} {suffix}{AI_PROMPT}"
                    for suffix, user_prompt in zip(df_suffixes['control'], df_suffixes['goals'])
                ]
                experiment = AnthropicCompletionExperiment([args.model_name], messages,
                                                max_tokens_to_sample=[64],
                                                  **gen_config_override
                                                  )
            else:
                raise NotImplementedError('unsupported API')

            experiment.run()
            df_answers = experiment.get_table(get_all_cols = True)
            df_answers['model_suffix'] = model_suffix
            df_answers['system_prompt'] = scenario
            df_answers['date'] = get_datetime()
            if isinstance(df_answers['response'][0], list):
                if df_answers['response'].apply(lambda x: len(x)>1).any():
                    print(f'[ERROR] Multiple answers received. Considering the first one only. All anwers saved in the "response_backup" column.')
                    df_answers['response_backup'] = df_answers['response']
                df_answers['response'] = df_answers['response'].apply(lambda x: x[0])
            df_answers['answer_generated'] = df_answers['response'].str.extract(r'(\d{'+str(desired_size)+'})')
            # match suffix with answer
            if args.api_name == 'openai':
                df_answers['full_prompt'] = df_answers['messages'].apply(lambda x: x[1]['content']).astype(str)
            elif args.api_name == 'anthropic':
                df_answers['full_prompt'] = df_answers['prompt'].str.replace(HUMAN_PROMPT, '')
                df_answers['full_prompt'] = df_answers['full_prompt'].str.replace(AI_PROMPT, '')
                if system_prompt:
                    df_answers['full_prompt'] = df_answers['full_prompt'].str.replace(system_prompt, '')
            else:
                raise NotImplementedError('Should implement how to handle prompt')
            df_answers = df_answers.merge(df_suffixes, on='full_prompt', how='left', suffixes=[None, 'suffix_'])
            df_answers['answer_target'] = df_answers['number']
            # export individual answers
            path_answers = args.export_csv.replace('retrieval_rate', 'answers')
            save_csv(df_answers, path_answers)

            # compute stats
            n_total = df_answers.shape[0]
            n_reject_total = df_answers['answer_generated'].isna().sum()
            n_ok_total = (df_answers['answer_target'] == df_answers['answer_generated']).sum()
            nb_answers = n_total - n_reject_total
            print(
                f'[{i+1}/{args.n_gen}] Retrieval rate for the {scenario} scenario on model {args.model_name}: {n_ok_total / nb_answers * 100:.2f}% ({n_ok_total}/{nb_answers}), rejection rate={n_reject_total / n_total * 100:.2f}% ({n_reject_total}/{n_total}), based on {len(list_adv_suffix)} adv suffixes.')
            df_stats = pd.DataFrame([{
                    'model_suffix': model_suffix,
                    'model': args.model_name,
                    'system_prompt': scenario,
                    'retrieval_rate': n_ok_total / nb_answers,  # % of correct answers
                    'no_answer_rate': n_reject_total / n_total,  # rate of no answer
                    'nb_suffixes': len(list_adv_suffix),
                    'nb_generation': n_total,
                    'nb_answers': nb_answers,
                    'nb_correct_answers': n_ok_total,
                    'nb_no_answers': n_reject_total,
                    **gen_config_override,
                    'date': get_datetime(),
                }])
            save_csv(df_stats, args.export_csv)
            df_all = pd.concat([df_all, df_stats], ignore_index=True)
        # compute final stats across N gens
        n_ok_total = df_all['nb_correct_answers'].sum()
        nb_answers = df_all['nb_answers'].sum()
        n_reject_total = df_all['nb_no_answers'].sum()
        n_total = df_all['nb_generation'].sum()
        print(
            f'[FINAL] ==> Retrieval rate for the {scenario} scenario on model {args.model_name}: {n_ok_total / nb_answers * 100:.2f}% ({n_ok_total}/{nb_answers}), rejection rate={n_reject_total / n_total * 100:.2f}% ({n_reject_total}/{n_total}), based on {len(list_adv_suffix)} adv suffixes.')


if __name__ == "__main__":
    main()
