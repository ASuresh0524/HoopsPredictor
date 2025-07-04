import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerStatsDataset(Dataset):
    """
    Dataset class for player statistics time series data.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        target_cols: List[str]
    ):
        """
        Dataset for player statistics sequences.
        
        Args:
            data (pd.DataFrame): Player game statistics
            sequence_length (int): Number of games in each sequence
            target_cols (List[str]): Statistics to predict
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_cols = target_cols
        
        # Create sequences for each player
        self.sequences = []
        for player_id in data['Player_ID'].unique():
            player_data = data[data['Player_ID'] == player_id]
            if len(player_data) >= sequence_length:
                for i in range(len(player_data) - sequence_length):
                    self.sequences.append({
                        'input': player_data.iloc[i:i+sequence_length],
                        'target': player_data.iloc[i+sequence_length][target_cols].values
                    })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Convert input sequence to tensor
        input_tensor = torch.FloatTensor(
            sequence['input'][self.target_cols].values
        )
        
        # Convert target to tensor
        target_tensor = torch.FloatTensor(sequence['target'])
        
        return input_tensor, target_tensor

class PlayerStatsLSTM(nn.Module):
    """
    LSTM model for predicting player statistics.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        """
        LSTM model for player statistics prediction.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, input_size)
        """
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # Use last sequence step
        return predictions

class PlayerStatsPredictor:
    """
    Main class for training and predicting player statistics using LSTM.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the predictor with configuration settings.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = self.config['models']['time_series']['lstm']['sequence_length']
        self.hidden_size = self.config['models']['time_series']['lstm']['hidden_size']
        self.num_layers = self.config['models']['time_series']['lstm']['num_layers']
        self.dropout = self.config['models']['time_series']['lstm']['dropout']
        self.learning_rate = self.config['models']['time_series']['lstm']['learning_rate']
        self.batch_size = self.config['models']['time_series']['lstm']['batch_size']
        self.epochs = self.config['models']['time_series']['lstm']['epochs']
        
        self.target_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        self.model = None
        
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[PlayerStatsDataset, PlayerStatsDataset, PlayerStatsDataset]:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            data (pd.DataFrame): Player statistics data
            
        Returns:
            Tuple[PlayerStatsDataset, PlayerStatsDataset, PlayerStatsDataset]:
                Training, validation, and test datasets
        """
        # Sort by date
        data = data.sort_values('Date')
        
        # Split data
        train_end = pd.to_datetime(self.config['data']['train_end_date'])
        val_end = pd.to_datetime(self.config['data']['val_end_date'])
        
        train_data = data[data['Date'] <= train_end]
        val_data = data[
            (data['Date'] > train_end) &
            (data['Date'] <= val_end)
        ]
        test_data = data[data['Date'] > val_end]
        
        # Create datasets
        train_dataset = PlayerStatsDataset(
            train_data,
            self.sequence_length,
            self.target_cols
        )
        val_dataset = PlayerStatsDataset(
            val_data,
            self.sequence_length,
            self.target_cols
        )
        test_dataset = PlayerStatsDataset(
            test_data,
            self.sequence_length,
            self.target_cols
        )
        
        return train_dataset, val_dataset, test_dataset
        
    def train(
        self,
        train_dataset: PlayerStatsDataset,
        val_dataset: PlayerStatsDataset
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            train_dataset (PlayerStatsDataset): Training dataset
            val_dataset (PlayerStatsDataset): Validation dataset
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Initialize model
        self.model = PlayerStatsLSTM(
            input_size=len(self.target_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_losses.append(loss.item())
            
            # Record losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )
        
        return history
        
    def predict(
        self,
        sequence: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Predict player statistics for the next game.
        
        Args:
            sequence (pd.DataFrame): Recent game statistics
            
        Returns:
            Dict[str, float]: Predicted statistics
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if len(sequence) < self.sequence_length:
            raise ValueError(
                f"Sequence length must be at least {self.sequence_length}"
            )
        
        # Prepare input sequence
        input_sequence = sequence.tail(self.sequence_length)[self.target_cols].values
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Convert to dictionary
        prediction_dict = {
            stat: pred.item()
            for stat, pred in zip(self.target_cols, prediction[0])
        }
        
        return prediction_dict
        
    def evaluate(
        self,
        test_dataset: PlayerStatsDataset
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_dataset (PlayerStatsDataset): Test dataset
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each statistic
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics for each statistic
        metrics = {}
        for i, stat in enumerate(self.target_cols):
            stat_preds = predictions[:, i]
            stat_actuals = actuals[:, i]
            
            mae = np.mean(np.abs(stat_preds - stat_actuals))
            rmse = np.sqrt(np.mean((stat_preds - stat_actuals) ** 2))
            correlation = np.corrcoef(stat_preds, stat_actuals)[0, 1]
            
            metrics[stat] = {
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation
            }
        
        return metrics
        
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_cols': self.target_cols,
            'config': self.config
        }, path)
        
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """
        Load a saved model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        checkpoint = torch.load(path)
        
        self.target_cols = checkpoint['target_cols']
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = PlayerStatsLSTM(
            input_size=len(self.target_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Model loaded from {path}") 