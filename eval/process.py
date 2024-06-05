import json

with open('../data/AQA/test.jsonl') as iFile:
    for row in iFile:
        data = json.loads(row)
        print(data['answer'])
