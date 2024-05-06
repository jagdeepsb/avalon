import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from src.game.utils import Role
from src.utils.constants import (
    MODELS_DIR,
    RES_TRAIN_BELIEF_DATASET_PATH,
    RES_VAL_BELIEF_DATASET_PATH,
    SPY_TRAIN_BELIEF_DATASET_PATH,
    SPY_VAL_BELIEF_DATASET_PATH,
)
from src.datasets.belief_dataset import BeliefDataset
from src.models.belief_predictor import BeliefPredictor
from src.game.beliefs import all_possible_ordered_role_assignments


def validate(model: BeliefPredictor, val_dataset: BeliefDataset) -> float:
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
    return avg_loss
    

if __name__ == "__main__":
    
    EXPERIMENT_NAME = "spy_belief_16_30_10"
    is_spy = True
    
    # Dirs
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    model_save_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}.pt")
    
    # Problem spec
    roles = [
        Role.MERLIN,
        Role.RESISTANCE,
        Role.RESISTANCE,
        Role.SPY,
        Role.SPY,
    ]
    n_classes = len(all_possible_ordered_role_assignments(roles))
    
    # Load dataset
    if is_spy:
        train_data = np.load(SPY_TRAIN_BELIEF_DATASET_PATH)
        train_game_histories, train_belief_distributions = train_data["game_histories"], train_data["game_beliefs"]
        val_data = np.load(SPY_VAL_BELIEF_DATASET_PATH)
        val_game_histories, val_belief_distributions = val_data["game_histories"], val_data["game_beliefs"]
    else:
        train_data = np.load(RES_TRAIN_BELIEF_DATASET_PATH)
        train_game_histories, train_belief_distributions = train_data["game_histories"], train_data["game_beliefs"]
        val_data = np.load(RES_VAL_BELIEF_DATASET_PATH)
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
    n_epochs = 25
    
    lowest_val_loss = float("inf")
    for epoch in range(n_epochs):
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
        validation_loss = validate(model, val_dataset)
        if validation_loss < lowest_val_loss:
            lowest_val_loss = validation_loss
            torch.save(model, model_save_path)
        
        print(f"Epoch {epoch}: Avg Loss {avg_loss / len(train_loader)} Validation Loss {validation_loss}")
        
    # Validate
    # print(f"Validation Loss: {validate(model, val_dataset)}")
    print(f"Lowest Validation Loss: {lowest_val_loss}")