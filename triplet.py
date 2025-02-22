import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
MARGIN = 0.5

class SimpleTriplet:
    def __init__(self):
        # Detect hardware(gpu support if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = SentenceTransformer(MODEL_NAME).to(self.device)
        self.loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        print(f"Training on: {self.device}")

    def train(self, train_data):
        self.model.train()
        anchors, positives, negatives = zip(*train_data)
        
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for i in tqdm(range(0, len(anchors), BATCH_SIZE)):
                # Get batch
                batch_anchor = anchors[i:i+BATCH_SIZE]
                batch_pos = positives[i:i+BATCH_SIZE]
                batch_neg = negatives[i:i+BATCH_SIZE]
                
                # Get embeddings on correct device
                anchor_embs = self.model.encode(batch_anchor, 
                                               convert_to_tensor=True,
                                               device=self.device)
                pos_embs = self.model.encode(batch_pos,
                                            convert_to_tensor=True,
                                            device=self.device)
                neg_embs = self.model.encode(batch_neg,
                                            convert_to_tensor=True,
                                            device=self.device)
                
                # Calculate loss
                self.optimizer.zero_grad()
                loss = self.loss_fn(anchor_embs, pos_embs, neg_embs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(anchors):.4f}")

# Load triplet data
def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return [(item['Anchor'], item['Positive'], item['Negative']) for item in data]

if __name__ == "__main__":
    # Load and split data
    data = load_data("processed_data/triplets.json")
    train_data, _ = train_test_split(data, test_size=0.2)
    
    # Initialize and train model
    model = SimpleTriplet()
    model.train(train_data)
    
    # Save model
    model.model.save("./gpu_triplet_model")