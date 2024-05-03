import pandas as pd
import os
import json
import argparse

os.getcwd()
os.chdir('/home/dan/DeepLearning/mini_temporal/')

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Understanding in LLMs Training Script")

    parser.add_argument('--generation_file', type=str, required=True, help='file to clean')
    parser.add_argument('--sub_folder', type=str, required=False, default=3, help= 'What folder should the corrections be saved in?')
    parser.add_argument('--save_name', type=str, required=True, help ='typically no_t_no_cleaned etc.')
    return parser.parse_args()

args = parse_args()

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
        output_parts = entry['OUTPUT'].split('\nmodel')
        if len(output_parts) >= 2:
            question = output_parts[0][5:] # skip the 'user\n' bit
            prediction = output_parts[1]
            entry['QUESTION'] = question
            entry['PREDICTION'] = prediction

        else:
            entry['PREDICTION'] = entry['OUTPUT']  
            entry['QUESTION'] = entry['OUTPUT'] 
            print(f"Warning: OUTPUT format unexpected in entry with INDEX {entry['INDEX']}")
            # print(entry['OUTPUT'] )
            count+=1
            indexes.append(entry['INDEX'])
    print(count)
    return indexes 

#endregion
#region # RUN IT
# =============================================================================
# RUN IT
# =============================================================================
# Try Just Grabbing the Predicted Answer # @$@ change if not trained model
path = f'./generations/output/trained_{args.sub_folder}'

# LOAD FILE
df = load_funky_json(f'{path}/{args.generation_file}')

# FIX INDEXES
fix_index_discontinuities(df)

# GET PREDS AND QUESTIONS
extract_predictions_and_questions(df)

# SAVE
df = pd.DataFrame(df)
save_folder = f'./eval/eval_data/trained_{args.sub_folder}_cleaned' #@$@ add trained_ before subfolder fo trained files
df.to_json(f'{save_folder}/{args.save_name}.jsonl', lines=True, orient='records')