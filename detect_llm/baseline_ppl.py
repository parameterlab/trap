import os
import argparse
import random
import re
import torch
import datasets
import numpy as np
import pandas as pd
from openai import OpenAI
import anthropic
from tqdm import tqdm
#from torcheval.metrics.functional.text import perplexity
from utils import load_system_prompts, save_csv
from compute_results import load_template, load_model
from llm_attacks.minimal_gcg.string_utils import SuffixManager



DATASETS = ['writing', 'pubmed', 'wiki']
APIS = ['openai', 'anthropic']
MAX_TOKENS = 512


def load_writing():
    with open(f'data/datasets/writing/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    def process_prompt(prompt):
        # adapted to filter all the `[ XX ]`
        pattern = r"^\s*\[\s*[A-Za-z]{2}\s*\]\s*"
        prompt = re.sub(pattern, "", prompt)
        prompt = re.sub(pattern, "", prompt)
        return prompt
        #return prompt.replace('[ WP ]', '').replace('[ OT ]', '').replace('[ EU ]', '').replace('[ IP ]', '')
    def process_spaces(story):
        return story.replace(
            ' ,', ',').replace(
            ' .', '.').replace(
            ' ?', '?').replace(
            ' !', '!').replace(
            ' ;', ';').replace(
            ' \'', '\'').replace(
            ' ’ ', '\'').replace(
            ' :', ':').replace(
            '<newline>', '\n').replace(
            '`` ', '"').replace(
            ' \'\'', '"').replace(
            '\'\'', '"').replace(
            '.. ', '... ').replace(
            ' )', ')').replace(
            '( ', '(').replace(
            ' n\'t', 'n\'t').replace(
            ' i ', ' I ').replace(
            ' i\'', ' I\'').replace(
            '\\\'', '\'').replace(
            '\n ', '\n').strip()

    prompts = [prompt for prompt in prompts if 'nsfw' not in prompt and 'NSFW' not in prompt]
    prompts = [process_prompt(process_spaces(prompt)) for prompt in prompts]
    prompts = [prompt for prompt in prompts if len(prompt) > 15]
    not_clean = [prompt for prompt in prompts if prompt[0]=='[']
    if not_clean: print(f'Ignored {len(not_clean)} prompts not cleaned properly while loading')
    prompts = [prompt for prompt in prompts if prompt[0]!='[']
    prompts = [f'Write a short fictional story about what follows. {prompt}' for prompt in prompts if prompt[0]!='[']
    return prompts

def load_pubmed():
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train')
    questions = data['question']
    return questions

def load_wiki():
    data = datasets.load_dataset("aadityaubhat/GPT-wiki-intro", split='train')
    title = data['title']
    return [f'Write a 200 word wikipedia style introduction on {t}.' for t in title]


def load_prompts(dataset, n_prompts=1000, seed=42):
    """
    Load prompts of a dataset. Partially based on the code of DetectGPT
    https://github.com/eric-mitchell/detect-gpt/blob/main/custom_datasets.py
    """
    if dataset == 'writing':
        prompts = load_writing()
    elif dataset == 'pubmed':
        prompts = load_pubmed()
    elif dataset == 'wiki':
        prompts = load_wiki()
    else:
        raise ValueError(f'dataset {dataset} not supported')
    prompts = list(set(prompts))  # remove duplicates
    random.seed(seed)
    prompts = random.sample(prompts, k=n_prompts)
    return prompts


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
        logprobs=True,  # return logprobs of each token
        max_tokens=max_tokens,
        tool_choice=None,  # do not call fn, generate output
    )
    text = completion.choices[0].message.content
    logsprobs = [x.logprob for x in completion.choices[0].logprobs.content]
    ppl = np.exp(-np.mean(logsprobs))
    return text, ppl

def anthropic_sample_once(model, user_prompt, system_prompt, temperature=1.0, top_p=1.0, max_tokens=64):
    if system_prompt is None or user_prompt is None:
        raise ValueError('prompts cannot be None')
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
    )
    text = message.content[0].text
    ppl = None
    return text, ppl


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None, enable_ppl=True):
    if gen_config is None:
        gen_config = model.generation_config
    gen_config.max_new_tokens = MAX_TOKENS
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    outputs = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id,
                                return_dict_in_generate=True, output_scores=enable_ppl)
    output_ids = outputs.sequences[0][assistant_role_slice.stop:].cpu().numpy()
    if not enable_ppl:
        return output_ids, None
    # added ppl from https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
    # added: remove eos token if last to match openai API
    if output_ids[-1] == gen_config.eos_token_id:
        outputs.sequences = outputs.sequences[:, :-1]
        outputs.scores = outputs.scores[:-1]
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    logsprobs = (transition_scores[0]).cpu().numpy()
    if len(logsprobs) > len(output_ids):
        logsprobs = logsprobs[-len(output_ids):]
    if len(output_ids) - len(logsprobs) >= 2:
        raise RuntimeError('output_ids not same length as logsprobs')
    ppl = np.exp(-np.mean(logsprobs))
    return output_ids, ppl

def model_sample_once(user_prompt, model, tokenizer, model_name, system_prompt=None, temperature=1.0, top_p=1.0, enable_ppl=True, device='cuda:0'):
    adv_suffix, target = '', ' '  # be careful target should not be an empty string for correct generation (otherwise the [/INST] is lost) !
    conv_template = load_template(model_name=model_name, system_prompt=system_prompt)
    suffix_manager = SuffixManager(tokenizer=tokenizer,
              conv_template=conv_template,
              instruction=user_prompt,
              target=target,
              adv_string=adv_suffix)
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    gen_config = model.generation_config
    gen_config.max_new_tokens = MAX_TOKENS
    gen_config.temperature = temperature
    gen_config.top_p = top_p
    output_ids, ppl = generate(model, tokenizer, input_ids, suffix_manager._assistant_role_slice, gen_config=gen_config, enable_ppl=enable_ppl)
    text = tokenizer.decode((output_ids)).strip()
    return text, ppl


def compute_ppl(user_prompt, target, model, tokenizer, model_name, system_prompt=None, device='cuda:0'):
    adv_suffix = ''
    conv_template = load_template(model_name=model_name, system_prompt=system_prompt)
    suffix_manager = SuffixManager(tokenizer=tokenizer,
                                   conv_template=conv_template,
                                   instruction=user_prompt,
                                   target=target,
                                   adv_string=adv_suffix)
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    target_ids = input_ids.clone()
    if input_ids[-1] == model.generation_config.eos_token_id:
        target_ids[-1] = -100  # do not compute loss on eos token
    target_ids[:suffix_manager._target_slice.start] = -100  # do not compute loss on prompt token
    input_ids = input_ids.unsqueeze(0)
    target_ids = target_ids.unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids, attention_mask=torch.ones_like(input_ids).to(model.device))
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss  # computed on generated tokens (not prompt)

    ppl = torch.exp(neg_log_likelihood).cpu().item()
    if not np.isfinite(ppl):
        print('NA in perplexity computation')
        breakpoint()
    return ppl



def main():
    parser = argparse.ArgumentParser(description='Identification using perplexity.')
    parser.add_argument('goal', choices=['gen', 'eval'], help='What to do: either generate text (gen) or evaluate PPL of previously generated texts (eval).')
    parser.add_argument('--gen-csv', help='CSV of generated text to evaluate. Ignored if goal=gen.')
    parser.add_argument('--dataset', choices=DATASETS, help='Dataset used for the prompt.')
    parser.add_argument('--n-prompts', default=1000, type=int, help='Nb of prompts from the datasets.')
    parser.add_argument('--api', choices=APIS, default=None, help='API to use to generate text. None (default), use open model.')
    parser.add_argument("--model-name", required=True, help="Name of the model used to generate or evaluate texts.")
    parser.add_argument("--model-path", default=None, help="Path of the opensource model (only used if api=None).")
    parser.add_argument("--system-prompt", default='original', help="Name of the system prompt to use. Does NOT support 'all'. Default (None), load the default model system prompt.")
    parser.add_argument("--temperature", default=0.6, type=float, help="Temperature")
    parser.add_argument("--top_p", default=0.9, type=float, help="Top-p")
    parser.add_argument("--export-base-folder", default='.', help="base directory to export csv")
    parser.add_argument("--export-sub-folder", default=None, help="create a subdirectory to export csv")
    parser.add_argument("--eval-filename", default=None, help="export eval into a file")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    args = parser.parse_args()

    if args.goal == 'gen':
        if not args.dataset: raise ValueError('empty --dataset')
        # enable ppl compute at gen only for small models (otherwise we got cuda outofmemory errors)
        enable_ppl = '7B' in args.model_name
        if args.temperature != 1.0:
            print(f'Temperature: {args.temperature}')
        if args.top_p != 1.0:
            print(f'Top_p: {args.top_p}')
    else: # eval
        if not args.gen_csv: raise ValueError('empty --gen-csv')

    if args.goal == 'gen':
        prompts = load_prompts(dataset=args.dataset, n_prompts=args.n_prompts)
        print(f'{len(prompts)} prompts loaded from the {args.dataset} dataset.')
    elif args.goal == 'eval':
        # load csv of texts
        pd_gen = pd.read_csv(args.gen_csv)
        prompts = pd_gen['prompt'].to_list()
        print(f'{len(prompts)} generated texts loaded from the csv {args.gen_csv} .')

    # load model
    if not args.api:
        if not args.model_path:
            raise ValueError('should specify model-path if no api')
        model, tokenizer = load_model(args.model_path)

    # system prompt
    system_prompt = load_system_prompts(name=args.system_prompt, model_name=args.model_name, return_dict=False)
    if args.system_prompt != 'original':
        print(f'Scenario: {args.system_prompt}')

    data = []
    for i, prompt in enumerate(tqdm(prompts, desc=args.goal)):
        if args.goal == 'gen':
            # API models
            if args.api == 'openai':
                text, ppl = openai_sample_once(model=args.model_name, user_prompt=prompt, system_prompt=system_prompt,
                                   temperature=args.temperature, top_p=args.top_p, max_tokens=MAX_TOKENS)
            elif args.api == 'anthropic':
                #print("We cannot compute generated PPL with anthopic")
                text, ppl = anthropic_sample_once(model=args.model_name, user_prompt=prompt, system_prompt=system_prompt,
                                   temperature=args.temperature, top_p=args.top_p, max_tokens=MAX_TOKENS)
            # open models
            else:
                text = '' ; n_tries = 0
                while len(text) < 15 and n_tries < 10:
                    if n_tries > 1:
                        print(f'[{i}] retrying generation (only {len(text)} char generated)')
                        breakpoint()
                    text, ppl = model_sample_once(user_prompt=prompt, model=model, tokenizer=tokenizer, model_name=args.model_name, system_prompt=system_prompt,enable_ppl=enable_ppl)
                    n_tries += 1
                if len(text) < 15:
                    continue  # skip the generation of this text if it failed

            data.append({
                'index': i,
                'api': args.api,
                'model': args.model_name,
                'system_prompt': args.system_prompt,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'prompt': prompt,
                'ppl': ppl,
                'text': text,
            })
        elif args.goal == 'eval':
            text = pd_gen['text'][i]
            ppl = compute_ppl(user_prompt=prompt, target=text, model=model, tokenizer=tokenizer, model_name=args.model_name, system_prompt=system_prompt)
            data.append({
                'gen_index': i,
                'gen_api':  pd_gen['api'][i],
                'gen_model': pd_gen['model'][i],
                'gen_system_prompt': pd_gen['system_prompt'][i],
                'gen_temperature': pd_gen['temperature'][i],
                'gen_top_p': pd_gen['top_p'][i],
                'gen_ppl': pd_gen['ppl'][i],
                'gen_csv': args.gen_csv,
                'model_eval': args.model_name,
                'eval_ppl': ppl,
                'prompt': prompt,
                'text': text,
            })
        else:
            raise ValueError('goal error')

    df = pd.DataFrame(data)


    if args.goal == 'gen':
        print(f'[PPL] avg: {df["ppl"].mean():.3f} ;  std: {df["ppl"].std()} ; computed on {df.shape[0]} generations')
        # path of generated texts
        path = os.path.join(args.export_base_folder, 'results/baseline/ppl/', 'dataset_' + args.dataset,
                            'gen_model_' + args.model_name)
        if args.export_sub_folder:
            path = os.path.join(path, args.export_sub_folder)
        filename_gen = f"gen_texts_n{args.n_prompts}_system_prompt_{args.system_prompt}_temperature_{str(args.temperature)}_top_p_{str(args.top_p)}_seed{args.seed}.csv"
        path_csv_gen = os.path.join(path, filename_gen)
        save_csv(df, path_csv_gen)
    else:
        print(f'[PPL] avg: {df["eval_ppl"].mean():.3f} ;  std: {df["eval_ppl"].std()}')
        # path of eval texts
        path = os.path.dirname(args.gen_csv)
        if args.export_base_folder:
            path = os.path.join(args.export_base_folder, path)
        if args.eval_filename is None:
            filename_gen = os.path.basename(args.gen_csv)
            if 'gen_' in filename_gen:
                args.eval_filename = filename_gen.replace('gen_', 'eval_')
            else:
                args.eval_filename = f"eval_texts_seed{args.seed}.csv"
        path_csv_eval = os.path.join(path, args.eval_filename)
        save_csv(df, path_csv_eval)



if __name__ == '__main__':
    main()
