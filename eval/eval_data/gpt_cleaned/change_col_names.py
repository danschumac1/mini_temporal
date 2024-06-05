import pandas as pd
import os 
os.chdir('/home/dan/DeepLearning/mini_temporal/eval/eval_data/gpt_cleaned')

no = pd.read_json('gpt_no_cleaned1.jsonl', lines = True, orient='records')
rel = pd.read_json('gpt_rel_cleaned1.jsonl', lines = True, orient='records')
rand = pd.read_json('gpt_rand_cleaned1.jsonl', lines = True, orient='records')
wd = pd.read_json('gpt_wd_cleaned1.jsonl', lines = True, orient='records')

df_dict = {'no':no, 'rel':rel, 'rand':rand, 'wd':wd}

for k, df in df_dict.items():
    try:
        if 'output' in df.columns:
            df_dict[k] = df.rename({'output': 'PREDICTION'}, axis=1)
            print(f'CHANGED {k}')
            df_dict[k].to_json(f'gpt_{k}_cleaned.jsonl', lines=True,orient='records')
        else:
            print(f'Column "output" not found in {k}')
    except Exception as e:
        print(f'Error processing {k}: {e}') 
