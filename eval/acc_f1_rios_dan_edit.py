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
# CUDA_VISIBLE_DEVICES=1 python acc_f1_rios_dan_edit.py --pred_file /home/dan/mini_temporal/generations/output/TQE/baseline/gemma-7b/relevant_context/TQE_baseline_gemma-7b_relevant_context_evaluated.jsonl --actual_file "/home/dan/mini_temporal/clean_for_fatemeh/new_data/test.jsonl" > minify_results.jsonl

import pandas as pd
import string
import re
import argparse
import os
import json
#os.chdir('/home/dan/DeepLearning/mini_temporal/eval')

#endregion
#region # ARGPARSE
# =============================================================================
# ARGPARSE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

    parser.add_argument('--file', type=str, required=True, help='file to score')
    parser.add_argument('--sub_folder', type=str, required=False, default=3, help= 'What folder should the corrections be saved in?')
    return parser.parse_args()

#args = parse_args()

#endregion
#region # HELPER FUNCTIONS
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def exact_match_f1(pred, answer_list):
    #pred_words = set(pred.lower().split())  # Convert the prediction into a set of words for faster operations
    pred_words = set(re.findall(r"\w+", pred.lower().replace("the answer is",""))[:5])
    best_f1 = 0  # Initialize best F1 score

    for answer in answer_list:
        answer_words = set(re.findall(r"\w+", answer.lower()))
        #answer_words = set(answer.lower().split())  # Convert answer into a set of words
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
    pred = ' '.join(re.findall(r"\w+", pred.lower()))
    normalized_pred = pred.lower().translate(translator)

    for answer in answer_list:
        # Normalize each answer
        normalized_answer = answer.lower().translate(translator)
        # Check if the normalized answer is contained within the normalized prediction
        if normalized_answer in normalized_pred:
            return 1

    return 0

#endregion
#region # LOAD DATA
# =============================================================================
# LOAD DATA
# =============================================================================
# GET THE CORRECT ANSWERS
actual = pd.read_json('./eval_data/test.jsonl', lines=True)
actual = actual.to_dict(orient='list')[0]
actual = [ans.split('__or__') for ans in actual]
print(actual)

# GET THE PREDICTIONS
model_generations = pd.read_json(f'final_check.jsonl',  lines=True)

# print('\n\n', model_generations.columns,'\n\n')

preds = [pred for pred in model_generations['PREDICTION']]

f1s = []
contains = []
for pred, ans_list in zip(preds, actual):
    f1s.append(exact_match_f1(pred, ans_list))
    contains.append(contains_metric(pred, ans_list))

avg_f1 = sum(f1s) / len(f1s)
avg_contains = sum(contains) / len(contains)
avg_f1

print(json.dumps({'F1': avg_f1, 'ACC':avg_contains}))