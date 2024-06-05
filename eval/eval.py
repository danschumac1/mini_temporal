#region
"""
Created on 04/29/2024

@author: Dan Schumacher
"""
#endregion

#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import string
import argparse
import os
import json

os.chdir('/home/dan/mini_temporal/')

#endregion







#endregion
#region # SET UP DATA       
# =============================================================================
# SET UP DATA       
# =============================================================================
# =============================================================================
# SET UP DATA       
# =============================================================================# =============================================================================
# SET UP DATA       
# =============================================================================# =============================================================================
# SET UP DATA       
# =============================================================================# =============================================================================
# SET UP DATA       
# =============================================================================

#endregion
#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import os
import json

os.getcwd()

os.chdir('/home/dan/mini_temporal')
import argparse

#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
# COMMAND-LINE ARGUMENTS
# def parse_args():
#     parser = argparse.ArgumentParser(description="Cleaning jsonl files")
#     parser.add_argument('--file', type=str, required=True, help='dataset to clean')
#     parser.add_argument('--skips', type=int, default=7, help='how many lines to skip')
#     parser.add_argument('--save_name', type=str, required=True, help='What to call output jsonl file')
#     parser.add_argument('--output_key',type=str, default='OUTPUT')
#     return parser.parse_args()

# args = parse_args()

#endregion
#region # HELPER FUNCTIONS
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_funky_json(file_path, skips=7):
    data = []
    with open(file_path, 'r') as file:
        # Skip the first 7 lines
        for _ in range(skips):
            next(file)
        
        # Process remaining lines
        for line in file:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} - Line skipped")
        return data

def check_index_continuity(data):
    """
    Function to check if the indices in the dataset are continuous or if there are breaks in the sequence.
    """
    previous_index = None
    discontinuities = []

    for entry in data:
        current_index = entry['INDEX']
        if previous_index is not None and current_index != previous_index + 1:
            discontinuities.append([previous_index, current_index])
        previous_index = current_index
    
    hole = discontinuities[0][0]
    return hole

def fix_index_discontinuities(data):
    """
    This function receives a list of dictionaries with an 'INDEX' key and adjusts indices to ensure continuity.
    It modifies the input data in-place.

    Parameters:
    data (list): A list of dictionaries each containing an 'INDEX' key.

    Returns:
    list: A list of the corrected indices if any corrections were made.
    """
    if not data or 'INDEX' not in data[0]:
        return []  # return early if data is empty or not in the expected format

    fixed_indices = []
    previous_index = data[0]['INDEX'] - 1  # set starting point correctly assuming the first index is correct

    for i, entry in enumerate(data):
        if entry['INDEX'] != previous_index + 1:
            corrected_index = previous_index + 1
            entry['INDEX'] = corrected_index
            fixed_indices.append(corrected_index)
        previous_index = entry['INDEX']

    return fixed_indices

def extract_predictions_and_questions(data):
    """
    Extracts predictions and questions from the 'OUTPUT' field in the data dictionaries.

    Parameters:
    data (list): A list of dictionaries with an 'OUTPUT' key.

    Returns:
    None: Modifies the dictionaries in the list to include 'PREDICTION' and 'QUESTION' keys.
    """
    count=0
    indexes = []
    for entry in data:
        output_parts = entry[args.output_key].split('\nmodel')
        for part in output_parts:
            part = part.replace(' ',' ')
            part = part.replace('\u2581',' ')

        if len(output_parts) >= 2:
            question = output_parts[0][5:] # skip the 'user\n' bit
            question = question.replace(' ',' ')
            question = question.replace('\u2581',' ')

            prediction = output_parts[1]
            prediction = prediction.replace(' ',' ')
            prediction = prediction.replace('\u2581',' ')

            entry['QUESTION'] = question            
            entry['PREDICTION'] = prediction

        else:
            entry['PREDICTION'] = entry[args.output_key].replace(' ',' ').replace('\u2581',' ')
            entry['QUESTION'] = entry[args.output_key].replace(' ',' ').replace('\u2581',' ')
            print(f"Warning: OUTPUT format unexpected in entry with INDEX {entry['INDEX']}")
            # print(entry['OUTPUT'] )
            count+=1
            indexes.append(entry['INDEX'])
    print(count)
    return indexes 

# data = load_funky_json(f'./gen_output/{args.file}', args.skips)
data = load_funky_json('./generations/trial_gen.jsonl')
data
# /home/dan/mini_temporal/generations/trial_gen.jsonl
fix_index_discontinuities(data)

# NIT
extract_predictions_and_questions(data)

save_folder = './gen_final'
save_name = args.save_name

data_df = pd.DataFrame(data)
data_df.to_json(f'{save_folder}/{save_name}.jsonl', lines=True, orient='records')
#endregion





















#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

    parser.add_argument('--file', type=str, required=True, help='file to score')
    parser.add_argument('--sub_folder', type=str, required=False, default=3, help='What folder should the corrections be saved in?')
    return parser.parse_args()

#endregion

#region # HELPER FUNCTIONS
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def exact_match_f1(pred, answer_list):
    pred_words = set(pred.lower().split())  # Convert the prediction into a set of words for faster operations
    best_f1 = 0  # Initialize best F1 score

    for answer in answer_list:
        answer_words = set(answer.lower().split())  # Convert answer into a set of words
        TP = len(answer_words.intersection(pred_words))
        FP = len(pred_words.difference(answer_words))
        FN = len(answer_words.difference(pred_words))
        
        if TP == 0:
            f1 = 0
        else:
            prec = TP / (TP + FP) if TP + FP > 0 else 0
            rec = TP / (TP + FN) if TP + FN > 0 else 0
            if (prec + rec) > 0:
                f1 = 2 * ((prec * rec) / (prec + rec))
            else:
                f1 = 0

        if f1 > best_f1:
            best_f1 = f1

    return best_f1

def contains_metric(pred, answer_list):
    """
    Checks if any answer in the list is contained within the prediction after removing punctuation
    and converting to lowercase.

    Parameters:
    - pred (str): The prediction string to be evaluated.
    - answer_list (list of str): A list of answer strings against which the prediction is evaluated.

    Returns:
    - bool: True if any answer is contained within the prediction, False otherwise.
    """
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    normalized_pred = pred.lower().translate(translator)

    for answer in answer_list:
        # Normalize each answer
        normalized_answer = answer.lower().translate(translator)
        # Check if the normalized answer is contained within the normalized prediction
        if normalized_answer in normalized_pred:
            return 1

    return 0

# def evaluate2(preds, answers):
#     f1s = []
#     contains = []
#     for pred, answer in zip(preds, answers):
#         print('PRED:', pred)
#         print('ANS:', answer)
#         em = exact_match_f1(pred, answer)
#         print(em)
#         print()

def evaluate(k, df):
    f1s = []
    contains = []
    for pred, ans_list in zip(df, actual):
        f1s.append(exact_match_f1(pred, ans_list))
        contains.append(contains_metric(pred, ans_list))
    
    avg_f1 = sum(f1s) / len(f1s)
    avg_contains = sum(contains) / len(contains)

    print(k)
    print('f1:', avg_f1)
    print('acc:', avg_contains)
    print()

#endregion

if __name__ == "__main__":  # @$@
    args = parse_args()  # @$@

    #region # LOAD DATA
    # =============================================================================
    # LOAD DATA
    # =============================================================================
    # GET THE CORRECT ANSWERS
    actual = pd.read_json('./eval_data/test.jsonl', lines=True)
    actual = actual.to_dict(orient='list')[0]
    actual = [ans.split('__or__') for ans in actual]

    # GET THE PREDICTIONS
    # print('\n\n', f'./eval_data/{args.sub_folder}/{args.file}', '\n\n')

    model_generations = pd.read_json(f'./eval_data/{args.sub_folder}/{args.file}', lines=True)

    # print('\n\n', model_generations.columns, '\n\n')

    preds = [pred for pred in model_generations['PREDICTION']]

    f1s = []
    contains = []
    for pred, ans_list in zip(preds, actual):
        f1s.append(exact_match_f1(pred, ans_list))
        contains.append(contains_metric(pred, ans_list))

    avg_f1 = sum(f1s) / len(f1s)
    avg_contains = sum(contains) / len(contains)
    avg_f1

    print(json.dumps({'NAME': args.file[:-6], 'F1': avg_f1, 'ACC': avg_contains}))  # @$@
