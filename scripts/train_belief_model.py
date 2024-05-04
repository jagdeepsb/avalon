import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.game.utils import Role
from src.utils.constants import (
    MODELS_DIR,
    TRAIN_BELIEF_DATASET_PATH,
    VAL_BELIEF_DATASET_PATH,
)
from src.datasets.belief_dataset import BeliefDataset
from src.models.belief_predictor import BeliefPredictor
from src.game.beliefs import num_possible_assignments


def validate(model: BeliefPredictor, val_dataset: BeliefDataset):
    """
    Validate the model on the validation dataset
    """
    
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            if i == 0:
                print(y_pred[0])
                print(y[0])
                print()
            
            avg_loss += loss.item()
    avg_loss /= len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    

if __name__ == "__main__":
    
    EXPERIMENT_NAME = "belief_tf_16_30_10"
    
    # Problem spec
    roles = [
        Role.MERLIN,
        Role.RESISTANCE,
        Role.RESISTANCE,
        Role.SPY,
        Role.SPY,
    ]
    n_classes = num_possible_assignments(roles)
    
    # Load dataset
    train_data = np.load(TRAIN_BELIEF_DATASET_PATH)
    train_game_histories, train_belief_distributions = train_data["game_histories"], train_data["game_beliefs"]
    val_data = np.load(VAL_BELIEF_DATASET_PATH)
    val_game_histories, val_belief_distributions = val_data["game_histories"], val_data["game_beliefs"]

    print(f"Train game history shape: {train_game_histories.shape}")
    print(f"Train belief shape: {train_belief_distributions.shape}")
    print(f"Val game history shape: {val_game_histories.shape}")
    print(f"Val belief shape: {val_belief_distributions.shape}")
    
    # Create datasets
    train_dataset = BeliefDataset(train_game_histories, train_belief_distributions)
    val_dataset = BeliefDataset(val_game_histories, val_belief_distributions)
    
    # Train model
    model = BeliefPredictor(
        encoding_dim=16,
        n_classes=n_classes,
        feature_dim=train_game_histories.shape[-1],
    )
    model.train()
    
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    n_epochs = 50
    
    for epoch in range(n_epochs):
        avg_loss = 0
        for i, (x, y) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        print(f"Epoch {epoch}: Avg Loss {avg_loss / len(train_loader)}")
        
    # Save model
    model_save_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}.pt")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    torch.save(model, model_save_path)
        
    # Validate
    validate(model, val_dataset)