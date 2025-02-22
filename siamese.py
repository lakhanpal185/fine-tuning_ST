# File: simple_siamese.py
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2' 
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
MARGIN = 0.5

class SimpleSiamese:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.loss_fn = nn.CosineEmbeddingLoss(margin=MARGIN)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
    def train(self, train_data):
        self.model.train()
        anchors, statements, labels = zip(*train_data)
        
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for i in tqdm(range(0, len(anchors), BATCH_SIZE)):
                batch_anchors = anchors[i:i+BATCH_SIZE]
                batch_statements = statements[i:i+BATCH_SIZE]
                batch_labels = torch.tensor(labels[i:i+BATCH_SIZE], dtype=torch.float32)
                
                # Get embeddings
                anchor_embs = self.model.encode(batch_anchors, convert_to_tensor=True)
                statement_embs = self.model.encode(batch_statements, convert_to_tensor=True)
                anchor_embs.requires_grad = True
                statement_embs.requires_grad = True
                
                # Calculate loss
                self.optimizer.zero_grad()
                loss = self.loss_fn(anchor_embs, statement_embs, batch_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(anchors):.4f}")

    def evaluate(self, test_data):
        self.model.eval()
        anchors, statements, labels = zip(*test_data)
        
        pro_sims = []
        con_sims = []
        
        with torch.no_grad():
            for anchor, statement, label in test_data:
                anchor_emb = self.model.encode(anchor)
                statement_emb = self.model.encode(statement)
                similarity = util.cos_sim(anchor_emb, statement_emb).item()
                
                if label == 1:
                    pro_sims.append(similarity)
                else:
                    con_sims.append(similarity)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.hist(pro_sims, alpha=0.5, label='Pro Statements')
        plt.hist(con_sims, alpha=0.5, label='Con Statements')
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig("similarity_distribution.png")
        
        # Calculate accuracy
        threshold = np.median(pro_sims + con_sims)
        accuracy = (np.mean([s > threshold for s in pro_sims]) + 
                   np.mean([s <= threshold for s in con_sims])) / 2
        print(f"Validation Accuracy: {accuracy:.2%}")

# Load data
def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
        
    return [(item['X1'], item['X2'], item['Label']) for item in data]

# Main execution
if __name__ == "__main__":
    # Load and split data
    data = load_data("processed_data/siamese_pairs.json")
    train_data, test_data = train_test_split(data, test_size=0.2)
    
    # Initialize and train model
    model = SimpleSiamese()
    model.train(train_data)
    
    # Evaluate
    model.evaluate(test_data)