from argparse import Namespace
from oemer import ete
from oemer import layers
import os
import json

'''path_to_data = "/home/ubuntu/KeyGenius/Music_Data/Scores"
out_dir = "/home/ubuntu/KeyGenius/Music_Data/output"
os.makedirs(out_dir, exist_ok=True)

all_results = {}

for file in os.listdir("/KeyGenius/Music_Data/Scores"):
    if file.startswith('.'):
        continue
    print(f"Processing: {file}")

ete.extract(Namespace(
        img_path=os.path.join(path_to_data, file), 
        output_path=out_dir,
        use_tf=True,
        save_cache=False, 
        without_deskew=True
    ))
    
    notes = layers.get_layer('notes')
    all_results[file] = [[int(x) for x in n.bbox] for n in notes if n.bbox]

with open("/home/ubuntu/KeyGenius/Music_Data/coordinates.json", "w") as f:
    json.dump(all_results, f)
'''


path_to_data = "/Users/anandkashyap/Documents/GitHub/KeyGenius/Backend/Music_Data/Scores"

# Get all files and sort them
files = [f for f in os.listdir(path_to_data) if not f.startswith('.')]
files.sort()

print(f"Total files to process: {len(files)}")
print("\nProcessing order:")
for i, file in enumerate(files, 1):
    print(f"{i}. {file}")