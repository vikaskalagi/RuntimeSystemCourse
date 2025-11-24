import json
import glob

files = glob.glob("**/*.json", recursive=True)

data = []
for f in files:
    with open(f) as fh:
        data.append(json.load(fh))

print("\n=== MetaMorph Comparison ===\n")
for item in data:
    print(f"Language: {item['language']}")
    print(f"Execution time: {item['execution_time_ms']} ms")
    print(f"Return type: {item['type_info']['return_type']}")
    print("---")


