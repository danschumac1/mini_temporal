import os
os.chdir('/home/dan/DeepLearning/mini_temporal/generations/output/gpt')
import pandas as pd

def fix_length(file_loc):
    df = pd.read_json(file_loc)
    df = df.iloc[:75]
    return df

files = [
    'output_correct_context1.json',
    'output_irrelevant_context1.json',
    'output_simple1.json',
    'output_wrong_date_context1.json'
    ]

df_dict = {}
for fil in files:
    df_dict[fil] = fix_length(fil)


for k, df in df_dict.items():
     k =''.join(k.split('1'))
     df.to_json(k, lines=True, orient='records')


