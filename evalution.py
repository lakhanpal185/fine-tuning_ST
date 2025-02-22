import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import numpy as np

model = SentenceTransformer('./triplet_model')

# Load base model for comparison
base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load processed data for test data from timchen0618/Kialo
dataset = load_dataset("test_data/test_triplets.json", split="validation")


# Function to compute similarities
def evaluate_model(model, dataset, model_name):
    pro_sims = []
    con_sims = []
    
    for sample in dataset:
        # Encode texts
        claim_emb = model.encode(sample['anchor'], convert_to_tensor=True)
        pro_emb = model.encode(sample['positive'], convert_to_tensor=True)
        con_emb = model.encode(sample['negative'], convert_to_tensor=True)
        
        # Compute similarities
        pro_sim = util.cos_sim(claim_emb, pro_emb).item()
        con_sim = util.cos_sim(claim_emb, con_emb).item()
        
        pro_sims.append(pro_sim)
        con_sims.append(con_sim)
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pro_sims, label='Pro Similarity', fill=True)
    sns.kdeplot(con_sims, label='Con Similarity', fill=True)
    plt.title(f'Cosine Similarity Distribution ({model_name})')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    return pro_sims, con_sims

# Evaluate both models
print("Evaluating Base Model...")
base_pro, base_con = evaluate_model(base_model, dataset, "Base Model")

print("\nEvaluating Fine-tuned Model...")
ft_pro, ft_con = evaluate_model(model, dataset, "Fine-tuned Model")

# Statistical comparison
print(f"\nBase Model - Pro avg: {np.mean(base_pro):.3f}, Con avg: {np.mean(base_con):.3f}")
print(f"Fine-tuned - Pro avg: {np.mean(ft_pro):.3f}, Con avg: {np.mean(ft_con):.3f}")
