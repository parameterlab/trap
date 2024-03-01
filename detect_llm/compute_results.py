"""
Generate CSV of goal/target
"""
import argparse
import os
import random
import pandas as pd

import json
import re
from tqdm import tqdm

import numpy as np
import torch

from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template


from utils import create_parent_folder, load_suffixes, save_csv, get_datetime, load_system_prompts

# supported model names
MODEL_NAMES = ['llama-2', 'vicuna', 'guanaco']

def load_model(model_path, device='cuda:0'):
    model, tokenizer = load_model_and_tokenizer(model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)
    model.requires_grad_(False)
    return model, tokenizer

def load_template(model_name, system_prompt=None):
    namesmatch = {
        'llama2-7B': 'llama-2',
        'llama2-13B': 'llama-2',
        'vicuna-7B': 'vicuna',
        'vicuna-13B': 'vicuna',
        'guanaco-7B': 'guanaco',
        'guanaco-13B': 'guanaco',
    }
    if model_name in namesmatch.keys():
        model_name = namesmatch[model_name]
    if not model_name in MODEL_NAMES:
        raise NotImplementedError(f'Template loading not supported for model: {model_name}.')
    conv_template = load_conversation_template(model_name, system=system_prompt)
    return conv_template


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None, no_warning=False):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    if gen_config.max_new_tokens > 50 and not no_warning:
        print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids[assistant_role_slice.stop:]


def compute_success_n_times(adv_suffix, num_target, user_prompt, target_str, model, tokenizer, model_name, system_prompt=None, n=10, verbose=0, device='cuda:0'):
    desired_size = len(num_target)
    target = target_str
    #target = f"Sure, here is a random string of {desired_size} digits: {num_target}"
    #user_prompt = f"Write a random string composed of {desired_size} digits. Your reply should only contain the random string."
    conv_template = load_template(model_name=model_name, system_prompt=system_prompt)
    suffix_manager = SuffixManager(tokenizer=tokenizer,
              conv_template=conv_template,
              instruction=user_prompt,
              target=target,
              adv_string=adv_suffix)
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64 #256
    n_ok, n_reject, answers = 0, 0, []
    for i in range(n):
        completion = tokenizer.decode((generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice, gen_config=gen_config, no_warning=True))).strip()
        # sometimes the answer is "formatted": e.g.: `{4211}`
        # so, we extract the fist N digits in the answer
        num_gen_search = re.search(r'(\d{'+str(desired_size)+'})', completion)
        num_gen = num_gen_search.group(1) if num_gen_search else None
        answers.append({'answer_generated': num_gen, 'answer_target': num_target, 'idx_gen': i, 'suffix': adv_suffix, 'completion': completion})
        # no answer or answer not valid
        if num_gen is None:
            n_reject += 1
            if verbose >= 2:
                print(f"Number extraction failed for: {completion}\n")
        # valid answer
        else:
            if num_gen == num_target:
                n_ok += 1
        if verbose >= 5:
            print(f"Num generated: {num_gen} (completion: {completion})")
    if verbose >= 1:
        print(f'* Nb success: {n_ok}/{n} = {100 * n_ok / n:.2f}%. Rejected: {n_reject}/{n} = {100 * n_reject / n:.2f}%')
    return n_ok, n_reject, n, answers


def main():
    parser = argparse.ArgumentParser(description="Compute retrieval rate from JSON suffixes.")
    parser.add_argument("-p", "--path-suffixes", required=True, help="Path to the folder with JSON files of suffixes")
    parser.add_argument("-t", "--suffix-step", default=None, type=int, help="Evaluate the suffix at a specific iteration. If None (default), evaluate at best iteration (lowest loss).")
    parser.add_argument("-m", "--model-path", required=True, help="Path to the model to use for generating")
    parser.add_argument("-o", "--model-name", choices=MODEL_NAMES, help="Name of the model. Template name.")
    parser.add_argument("-s", "--model-version", default=None, help="version of the model, ex 'Vicuna13B'")
    parser.add_argument("-f", "--export-csv", default=None,  help="Export to this file")
    parser.add_argument("-n", "--n-gen", default=10, type=int, help="Number of answers to generate for each suffix.")
    #parser.add_argument("-s", "--string-type", choices=['number', 'string'], help="Type of goal string.")
    parser.add_argument("-y", "--system-prompt", default=None, help="Name of the system prompt to use. 'all' tries all the available system prompts. Default (None), load the default model system prompt.")
    parser.add_argument("-g", "--gen-config-override", default=None, help="Override generation config with the provided values. Default (None), load the default model gen config.")
    parser.add_argument("-e", "--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("-d", "--device", default='cuda:0', help="Pytorch device.")
    parser.add_argument("-i", "--ignore-errors", action='store_true', help="Ignore suffixes with errors.")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose (at =2 print completions that failed to extract number).")
    args = parser.parse_args()

    model_suffix = re.search(r'/model_([^/]+)/', args.path_suffixes).group(1) if re.search(r'/model_([^/]+)/', args.path_suffixes) else args.path_suffixes

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df_suffixes = load_suffixes(args.path_suffixes, step=args.suffix_step)
    list_adv_suffix = df_suffixes['control'].to_list()
    list_number = df_suffixes['number'].to_list()
    list_user_prompt = df_suffixes['goals'].to_list()
    list_target = df_suffixes['targets'].to_list()

    # load model and override the gen config if set
    model, tokenizer = load_model(args.model_path, device=args.device)
    if args.gen_config_override:
        try:
            gen_config_override = json.loads(args.gen_config_override.replace("'", '"'))  # json requires double quotes
        except (ValueError, SyntaxError) as e:
            print("[ERROR] invalid json to override generation config")
            raise e
        model.generation_config.update(**gen_config_override)
    else:
        gen_config_override = {}
    system_prompts_dict = load_system_prompts(name=args.system_prompt, model_name=args.model_name)

    if not args.export_csv:
        args.export_csv = os.path.join(args.path_suffixes, f"retrieval_rate{'_system_prompts' if args.system_prompt else ''}{'_'+'_'.join(gen_config_override.keys()) if args.gen_config_override else ''}.csv")

    df, df_answers = pd.DataFrame(), pd.DataFrame()
    # for each suffix, generate n completion, and check if the target num is present
    for scenario, system_prompt in system_prompts_dict.items():
        if scenario != 'original':
            print(f'*** SCENARIO: {scenario} ***')
        n_ok_total, n_reject_total, n_total, answers_list = 0, 0, 0, []
        for adv_suffix, num_target, user_prompt, target_str in tqdm(zip(list_adv_suffix, list_number, list_user_prompt, list_target), desc='Suffixes'):
            if args.ignore_errors and pd.isna(num_target):
                continue
            n_ok, n_reject, n, answers = compute_success_n_times(adv_suffix=adv_suffix, num_target=num_target, user_prompt=user_prompt, target_str=target_str,
                                                        model=model, tokenizer=tokenizer, model_name=args.model_name,
                                                        system_prompt=system_prompt, n=args.n_gen, verbose=args.verbose, device=args.device)
            n_ok_total += n_ok
            n_reject_total += n_reject
            n_total += n
            answers_list = answers_list + answers
        if args.gen_config_override:
            print(f'Generation config: {gen_config_override}')
        nb_answers = n_total - n_reject_total
        print(
            f'==> Retrieval rate for the *{scenario}* scenario: {n_ok_total / nb_answers * 100:.2f}% ({n_ok_total}/{nb_answers}), rejection rate={n_reject_total / n_total * 100:.2f}% ({n_reject_total}/{n_total}), based on {len(list_adv_suffix)} adv suffixes.')
        df = pd.concat([df,
            pd.DataFrame([{
                'model_suffix': model_suffix,
                'model': args.model_version if args.model_version else args.model_name,  # eval model
                'system_prompt': scenario,
                'retrieval_rate': n_ok_total / nb_answers,  # % of correct answers
                'no_answer_rate': n_reject_total / n_total,  # rate of no answer
                'nb_suffixes': len(list_adv_suffix),
                'nb_generation': n_total,
                'nb_answers': nb_answers,
                'nb_correct_answers': n_ok_total,
                'nb_no_answers': n_reject_total,
                'seed': args.seed,
                **gen_config_override,
                'date': get_datetime(),
            }])
        ], ignore_index=True)
        # individual answers
        params_dict = {
            'model_suffix': model_suffix,
            'model': args.model_version if args.model_version else args.model_name,
            'system_prompt': scenario,
            'seed': args.seed,
            **gen_config_override,
            'date': get_datetime(),
        }
        answers_list = [{**params_dict, **a} for a in answers_list]
        df_answers = pd.concat([df_answers, pd.DataFrame(answers_list)], ignore_index=True)


    # export stats
    save_csv(df, args.export_csv)
    # export individual answers
    directory, filename = os.path.dirname(args.export_csv), os.path.basename(args.export_csv)
    if 'retrieval_rate' in filename:
        filename = filename.replace('retrieval_rate', 'answers')
    else:
        filename = 'answers_' + filename
    path_answers = os.path.join(directory, filename)

    save_csv(df_answers, path_answers)


if __name__ == "__main__":
    main()
