# fetch_guppy_data.py

import requests
import json
import csv

GUPPY_URL = "http://guppy-service/graphql"
PAGE_SIZE = 1000

def get_mapping(node_type):
    query = {"query": f"{{ _mapping {{ {node_type} }} }}"}
    res = requests.post(GUPPY_URL, json=query)
    fields = res.json()["data"]["_mapping"][node_type]
    return fields

def fetch_all_data(node_type, fields):
    all_data = []
    offset = 0

    while True:
        field_str = " ".join(fields)
        query = {
            "query": f"""{{
                {node_type}(offset: {offset}, first: {PAGE_SIZE}) {{
                    {field_str}
                }}
            }}"""
        }
        res = requests.post(GUPPY_URL, json=query)
        items = res.json()["data"][node_type]
        if not items:
            break
        all_data.extend(items)
        offset += PAGE_SIZE
        print(f"Fetched {len(all_data)} records from {node_type}...")

    return all_data

def save_to_json_csv(data, node_type):
    with open(f"{node_type}_data.json", "w") as f:
        json.dump(data, f, indent=2)

    if data:
        keys = data[0].keys()
        with open(f"{node_type}_data.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

def main():
    for node_type in ["case", "follow_up"]:
        fields = get_mapping(node_type)
        data = fetch_all_data(node_type, fields)
        save_to_json_csv(data, node_type)
        print(f"{node_type} saved as JSON and CSV")

if __name__ == "__main__":
    main()
