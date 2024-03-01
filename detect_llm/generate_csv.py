"""
Generate CSV of goal/target
"""
import argparse
import random
import string
import pandas as pd

from utils import create_parent_folder

def generate_random_string(string_length, string_type, lower_case_only=True):
    if string_type == 'string':
        # Define the set of ASCII letters (both lowercase and uppercase)
        ascii_letters = string.ascii_letters
        if lower_case_only:
            ascii_letters = ascii_letters[:26]
    elif 'number' in string_type:
        ascii_letters = string.digits
    else:
        raise NotImplementedError(f'String type not supported: {string_type}')
    # Generate a random string of the specified size
    random_string = ''.join(random.choice(ascii_letters) for _ in range(string_length))
    return random_string

def get_prompt_target(string_length, string_type, random_string):
    if string_type == 'number':
        if string_length == 1:
            user_prompt = f"Write a random digit. Your reply should only contain the random digit."
            target = f"Sure, here is a random digit: {random_string}"
        else:
            user_prompt = f"Write a random string composed of {string_length} digits. Your reply should only contain the random string."
            target = f"Sure, here is a random string of {string_length} digits: {random_string}"
    elif string_type == 'number_simpletarget':
        if string_length == 1:
            user_prompt = f"Write a random digit. Your reply should only contain the random digit."
            target = f"{random_string}"
        else:
            user_prompt = f"Write a random string composed of {string_length} digits. Your reply should only contain the random string."
            target = f"{random_string}"
    elif string_type == 'string':
        user_prompt = f"Write a random string composed of {string_length} lower case letters. Your reply should only contain the random string."
        target = f'Sure, here is a random string of {string_length} lowercase letters: {random_string}'
    else:
        raise NotImplementedError('String type not supported')
    return {'goal': user_prompt, 'target': target, 'string_target': random_string}



def main():
    parser = argparse.ArgumentParser(description="Generate goal and target string in CSV.")
    parser.add_argument("-f", "--export-csv", default=None,  help="Export to this file")
    parser.add_argument("-n", "--n-goals", default=100, type=int, help="Number of goal strings to generate (number of lines in the CSV).")
    parser.add_argument("-m", "--method", choices=['random', 'nll'], help="Method to choose the goal string.")
    parser.add_argument("-s", "--string-type", choices=['number', 'number_simpletarget', 'string'], help="Type of goal string.")
    parser.add_argument("-l", "--string-length", type=int, default=5, help="Length of the goal string.")
    parser.add_argument("-d", "--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if not args.export_csv:
        args.export_csv = f'data/method_{args.method}/type_{args.string_type}/str_length_{args.string_length}/prompt_goal_n{args.n_goals}_seed{args.seed}.csv'

    random.seed(args.seed)
    if args.method == 'random':
        target_string_list = [generate_random_string(string_length=args.string_length, string_type=args.string_type) for _ in range(args.n_goals)]
    else:
        raise NotImplementedError('Method not implemented')

    data = [ get_prompt_target(string_length=args.string_length, string_type=args.string_type, random_string=target_string_list[i]) for i in range(args.n_goals) ]
    df = pd.DataFrame(data)

    create_parent_folder(args.export_csv)
    df.to_csv(args.export_csv, index=False)

if __name__ == "__main__":
    main()
