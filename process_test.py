import json

triplets = []
pairs = []

sourc_file = "/home/lakhan/Desktop/IIT-BOMBAY/OWI/assignment/Kialo/kialo.test.jsonl"
output_file = "/home/lakhan/Desktop/IIT-BOMBAY/OWI/assignment/test_data/"
with open(sourc_file, "r", encoding="utf-8") as infile:

    for line in infile:

        data = json.loads(line.strip())
        
        if data.get("type", "").lower() == "binary":
            question = data.get("question", "")
            perspectives = data.get("perspectives", [])
            
            if len(perspectives) >= 2: #binary
                con = perspectives[0]
                pro = perspectives[1]
                
                triplets.append( {
                    "anchor": question,
                    "positive": pro,
                    "negative": con    
                })
                
                pairs.append({"X1": question, "X2": con, "Label": 0 })
                pairs.append({"X1": question, "X2": pro, "Label": 1 })


with open(output_file+"test_triplets.json", "w", encoding="utf-8") as triplet_file:
    json.dump(triplets, triplet_file, indent=4)

with open( output_file+"test_pairs.json", "w", encoding="utf-8") as pairs_file:
    json.dump(pairs, pairs_file, indent=4)

print("Processing complete. test file have been generated.")
