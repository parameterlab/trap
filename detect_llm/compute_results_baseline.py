"""
Generate CSV of empirical distribution of answers
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

from llm_attacks.minimal_gcg.string_utils import SuffixManager

from utils import save_csv, load_system_prompts

from compute_results import load_model, load_template, MODEL_NAMES, generate



def generate_n_times(desired_size, user_prompt, target, model, tokenizer, model_name, system_prompt=None, n=10, verbose=0, device='cuda:0'):
    adv_suffix = ''
    conv_template = load_template(model_name=model_name, system_prompt=system_prompt)
    suffix_manager = SuffixManager(tokenizer=tokenizer,
              conv_template=conv_template,
              instruction=user_prompt,
              target=target,
              adv_string=adv_suffix)
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    gen_config = model.generation_config
    gen_config.max_new_tokens = 64
    n_reject, answers = 0, []
    with tqdm(total=n) as pbar:
        while len(answers) < n:
            completion = tokenizer.decode((generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice, gen_config=gen_config, no_warning=True))).strip()
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
    parser = argparse.ArgumentParser(description="Compute retrieval rate from JSON suffixes.")
    parser.add_argument("--model-path", required=True, help="Path to the model to use for generating")
    parser.add_argument("--model-name", choices=MODEL_NAMES, help="Name of the model. Template name.")
    parser.add_argument("--model-version", default=None, help="version of the model, ex 'Vicuna13B'")
    parser.add_argument("--export-base-folder", default='.', help="base directory to export csv")
    parser.add_argument("--export-sub-folder", default=None, help="create a subdirectory to export csv")
    parser.add_argument("--n-gen", default=1000, type=int, help="Number of answers to sample.")
    parser.add_argument("--n-digits", default=4, type=int, help="Number of digits of the answer.")
    parser.add_argument("--system-prompt", default=None, help="Name of the system prompt to use. Does NOT support 'all'. Default (None), load the default model system prompt.")
    parser.add_argument("--gen-config-override", default=None, help="Override generation config with the provided values. Default (None), load the default model gen config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default='cuda:0', help="Pytorch device.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose (at =2 print completions that failed to extract number).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    desired_size = args.n_digits
    target = f"Sure, here is a random string of {desired_size} digits: "
    user_prompt = f"Write a random string composed of {desired_size} digits. Your reply should only contain the random string."

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
    scenario = args.system_prompt
    system_prompt = system_prompts_dict['original' if scenario is None else scenario]
    if scenario != 'original':
        print(f'Scenario: {scenario}')

    n_reject, n, answers_list = generate_n_times(desired_size=desired_size, user_prompt=user_prompt, target=target,
                                            model=model, tokenizer=tokenizer, model_name=args.model_name,
                                            system_prompt=system_prompt, n=args.n_gen, verbose=args.verbose,
                                            device=args.device)
    if args.gen_config_override:
        print(f'Generation config: {gen_config_override}')
    # individual answers
    df_answers = pd.DataFrame({'answer': answers_list})

    # export individual answers
    path = os.path.join(args.export_base_folder, 'results/baseline/answers_nosuffix/', args.model_version)
    if args.export_sub_folder:
        path = os.path.join(path, args.export_sub_folder)
    filename = f"answers_samples_{args.n_digits}digits{'_system_prompt_'+args.system_prompt if args.system_prompt else ''}{'_'+'_'.join([f'{key}_{value}' for key, value in gen_config_override.items()]) if args.gen_config_override else ''}_seed{args.seed}.csv"
    path_answers = os.path.join(path, filename)
    save_csv(df_answers, path_answers)


if __name__ == "__main__":
    main()
