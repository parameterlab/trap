import os
import glob
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd


def create_parent_folder(filename):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)


def load_suffixes_csv(path: str):
    """
    Load a single CSV file of suffixes
    """
    return pd.read_csv(path)


def load_suffixes(path, seed=None, step=None):
    """
    Load the suffixes as dataframe
    :param path: Path containing JSON files
    :param seed: Load only the suffixes of a specific random seed
    :param step: Load suffixes at a specific optimization step. Default (None), load the suffixes at the best iteration (lowest loss)
    """
    if '.csv' in path:
        if step is not None:
            raise NotImplementedError('CSV loading does not support step.')
        return load_suffixes_csv(path=path)

    files = glob.glob(os.path.join(path, "*.json"))
    if len(files) == 0:
        raise ValueError(f'Empty directory no JSON/CSV files in: {path}')
    if seed:
        files = [f for f in files if f'seed{seed}_' in f]  # filter filename with the seed
        if len(files) == 0:
            raise ValueError(f'No JSON/CSV files with seed: {seed}')
    files = [f for f in files if os.path.getsize(f) > 0]  # ignore empty files
    files = sorted(files, key=lambda x: "_".join(x.split('_')[:-1]))
    data = []
    for file in files:
        with open(file, 'r') as f:
            data_json = json.load(f)
        if step is None:
            data += data_json['best']
        else:
            n_steps = data_json['params']["n_steps"]
            eval_steps = data_json['params']['test_steps']
            control_init = data_json['params']['control_init']
            for i, control in enumerate(data_json["controls"]):
                #i_step = i % (1 + n_steps // eval_steps)
                if i % (1 + n_steps // eval_steps) == 0 and control != control_init:
                    raise RuntimeError('Error while parsing suffix JSON')
                if i % (1 + n_steps // eval_steps) == step // eval_steps:
                    i_goal = (i * eval_steps) // n_steps
                    data += [{
                        "goals": data_json['params']['goals'][i_goal],
                        "targets": data_json['params']['targets'][i_goal],
                        "control": control,
                        "loss": data_json['losses'][i],
                        "step": i % (1 + n_steps // eval_steps),
                    }]
    print(f'{len(data)} suffixes loaded from {len(files)} files.')
    if len(data) == 0:
        raise ValueError(f'No suffixes found in the JSON files in: {path}')
    for i,suffix in enumerate(data):
        for k,v in suffix.items():
            if type(v)==list and len(v) == 1:
                data[i][k] = v[0]
    str_length_search = re.search(r'\/str_length_(\d+)\/', path)
    if str_length_search:
        str_length = str_length_search.group(1)
    else:
        print(f'[INFO] String length not detected from suffix path (/str_length_XX/). Using 4 by default.')
        str_length = 4
    df = pd.DataFrame(data)
    df['number'] = df['targets'].str.extract(r'(\d{'+str(str_length)+'})')
    df['str_length'] = str_length
    if pd.isna(df['number']).sum() > 0:
        print(f"[ERROR] extracting targeted number: {pd.isna(df['number']).sum()} NA values!")
    return df


def load_system_prompts(name, model_name, path_prompts='data/system_prompts/scenario_prompts.json', return_dict=True):
    if 'llama-2' in model_name or 'llama2' in model_name: model_name = 'llama-2'
    if 'vicuna' in model_name: model_name = 'vicuna'
    if 'guanaco' in model_name: model_name = 'guanaco'
    if 'gpt-3.5' in model_name or 'gpt-4' in model_name: model_name = 'openai'
    if 'claude' in model_name: model_name = 'anthropic'
    if not name:
        if model_name in ['llama-2', 'vicuna', 'guanaco']:
            return {'original': None, }  # return None if None to use the default one loaded by fastchat
        else:
            name = 'original'
    with open(path_prompts, "r") as f:
        all_prompts = json.load(f)
    if model_name not in all_prompts.keys():
        raise ValueError(f'No model_name of {model_name} corresponding in scenario_prompts.json')
    system_prompts_dict = all_prompts[model_name]
    print(f'{len(system_prompts_dict)} system prompts loaded.')
    if name == 'all':
        if not return_dict:
            raise ValueError('Should return dict for all prompts')
        return system_prompts_dict
    if return_dict:
        return {name: system_prompts_dict[name],}
    else:
        return system_prompts_dict[name]



SUPPORTED_DISTANCES = ['exact', 'edit_distance', 'digit_distance', 'jaccard_index']

def distance_answer(answer: str, target: str, distance: str ='exact') -> int:
    """
    Compute distance between generated and target answers.
    :param answer:
    :param target:
    :param distance: 'exact' (true/false if exact string match), 'edit_distance' (Hamming distance), 'digit_distance' (sum of absolute diff of each digit)
    :return:
    """
    if pd.isnull([answer,target]).sum():
        return np.nan
    if distance == 'exact':
        return answer == target
    elif distance == 'edit_distance':
        # Hamming distance
        if len(answer) != len(target):
            raise ValueError("Strings must be of equal length.")
        return sum(char1 != char2 for char1, char2 in zip(answer, target))
    elif distance == 'digit_distance':
        # see https://www.cambridge.org/core/journals/mathematical-gazette/article/abs/digitdistance-mastermind/602804634243D602064C013B3A4BB706
        max_len = max(len(answer), len(target))
        answer, target = answer.zfill(max_len), target.zfill(max_len)
        # Calculate the sum of absolute differences of each digit
        return sum(abs(int(a) - int(b)) for a, b in zip(answer, target))
    elif distance == 'jaccard_index':
        # number of characters in common: card(intersection(A,B))/card(A U B)
        if len(answer) != len(target):
            raise ValueError("Strings must be of the same length")
        set1 = set(answer)
        set2 = set(target)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        # Jaccard Similarity Coefficient
        return len(intersection) / len(union)
    else:
        raise NotImplementedError(f'Distance {distance} not implemented')


def save_csv(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to a CSV file. If the file exists and has the same columns, append the data.
    If the columns do not match, merge the DataFrames and overwrite the file.

    :param df: Pandas DataFrame to be saved.
    :param path: Path to the CSV file.
    """
    create_parent_folder(path)
    #df.to_csv(path, index=False, mode='a', header=not os.path.isfile(path))
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        # Check if the columns match
        if set(df.columns) == set(existing_df.columns):
            # Append mode
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            # Merge and overwrite
            merged_df = pd.concat([existing_df, df], ignore_index=True)
            merged_df.to_csv(path, index=False)
    else:
        # Write new file
        df.to_csv(path, index=False)

def change_filename(path: str, new_filename: str) -> str:
    """
    Changes the filename in a given path.

    :param path: The original file path.
    :param new_filename: The new filename to replace the old one.
    :return: The path with the new filename.
    """
    dir_name, old_filename = os.path.split(path)
    new_path = os.path.join(dir_name, new_filename)
    return new_path

def get_datetime():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")