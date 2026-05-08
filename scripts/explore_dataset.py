import json
import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'iilb_parks.json')

if os.path.exists(data_path):
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total parks loaded: {len(data)}")
    if data:
        print(f"Sample record: {json.dumps(data[0], indent=2)}")
else:
    print(f"Dataset not found at {data_path}")
