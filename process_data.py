import json
import os
from tqdm import tqdm

def parse_discussion(file_path):
    """Parses the discussion file and extracts anchor, pro, and con arguments."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    anchor = lines[0].strip()
    pro_statements = []
    con_statements = []
    
    for line in lines[1:]:
        line = line.strip()
        if "Pro:" in line:
            pro_statements.append(line.split("Pro:")[-1].strip())
        elif "Con:" in line:
            con_statements.append(line.split("Con:")[-1].strip())
    
    return anchor, pro_statements, con_statements

def generate_siamese_pairs(anchor, pro_statements, con_statements):
    """Generates pairs for Siamese network (similar and dissimilar)."""
    siamese_pairs = []
    for pro in pro_statements:
        siamese_pairs.append({"X1": anchor, "X2": pro, "Label": 1})  # Agreeing pair
    for con in con_statements:
        siamese_pairs.append({"X1": anchor, "X2": con, "Label": 0})  # Disagreeing pair
    return siamese_pairs

def generate_triplets(anchor, pro_statements, con_statements):
    """Generates triplets for Triplet network."""
    triplets = []
    for pro in pro_statements:
        i = 15
        for con in con_statements:
            triplets.append({"Anchor": anchor, "Positive": pro, "Negative": con})
            i = i-1
            if i == 0:
                break
    return triplets

def process_data(input_folder, output_folder):
    """Processes all discussion files and saves Siamese pairs and Triplets."""
    siamese_data = []
    triplet_data = []
    
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            anchor, pro_statements, con_statements = parse_discussion(file_path)
            
            siamese_data.extend(generate_siamese_pairs(anchor, pro_statements, con_statements))
            triplet_data.extend(generate_triplets(anchor, pro_statements, con_statements))
    
    os.makedirs(output_folder, exist_ok=True)
    
    with open(os.path.join(output_folder, "siamese_pairs.json"), "w", encoding='utf-8') as f:
        json.dump(siamese_data, f, indent=4)
    
    with open(os.path.join(output_folder, "triplets.json"), "w", encoding='utf-8') as f:
        json.dump(triplet_data, f, indent=4)

# Example Usage
input_folder = "train"  # Folder containing discussion files
output_folder = "processed_data"  # Output folder for generated data
process_data(input_folder, output_folder)