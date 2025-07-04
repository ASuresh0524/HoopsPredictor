import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
        self.sequence_length = self.config['models']['player_predictor']['sequence_length']
        self.hidden_size = self.config['models']['player_predictor']['hidden_size']
        self.num_layers = self.config['models']['player_predictor']['num_layers']
        self.dropout = self.config['models']['player_predictor']['dropout']
        self.learning_rate = self.config['models']['player_predictor']['learning_rate']
        self.batch_size = self.config['models']['player_predictor']['batch_size']
        self.epochs = self.config['models']['player_predictor']['epochs']
        self.early_stopping_patience = self.config['models']['player_predictor']['early_stopping_patience']
        
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
        
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data (pd.DataFrame): Player statistics data
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences and target values
        """
        sequences = []
        targets = []
        
        # Sort by player and date
        data = data.sort_values(['PLAYER_NAME', 'Date'])
        
        # Group by player
        for _, player_data in data.groupby('PLAYER_NAME'):
            if len(player_data) < sequence_length + 1:
                continue
            
            # Get feature columns
            feature_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN']
            features = player_data[feature_cols].values
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                sequences.append(features[i:i + sequence_length])
                targets.append(features[i + sequence_length])
        
        return np.array(sequences), np.array(targets)

    def train(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the LSTM model on player statistics data.
        
        Args:
            data (pd.DataFrame): Player statistics data
            save_path (str, optional): Path to save the trained model
            
        Returns:
            Dict: Training history
        """
        # Prepare sequences
        X, y = self.prepare_sequences(data, self.sequence_length)
        
        # Scale data
        X_reshaped = X.reshape(-1, X.shape[-1])
        y_reshaped = y.reshape(-1, y.shape[-1])
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.feature_scaler.fit(X_reshaped)
        self.target_scaler.fit(y_reshaped)
        
        X_scaled = self.feature_scaler.transform(X_reshaped).reshape(X.shape)
        y_scaled = self.target_scaler.transform(y_reshaped).reshape(y.shape)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # Create model
        self.model = PlayerStatsLSTM(
            input_size=X.shape[-1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        history = {
            'loss': [],
            'val_loss': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i + self.batch_size]
                batch_y = y_tensor[i:i + self.batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X_tensor) / self.batch_size)
            history['loss'].append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'feature_scaler': self.feature_scaler,
                        'target_scaler': self.target_scaler
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return history

    def predict(
        self,
        player_history: pd.DataFrame,
        opponent: str,
        date: str
    ) -> Dict:
        """
        Predict player statistics for an upcoming game.
        
        Args:
            player_history (pd.DataFrame): Player's recent game history
            opponent (str): Opponent team name/code
            date (str): Game date
            
        Returns:
            Dict: Predicted statistics
        """
        # Prepare input sequence
        feature_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN']
        recent_games = player_history.sort_values('Date').tail(self.sequence_length)
        
        if len(recent_games) < self.sequence_length:
            raise ValueError(f"Not enough games in player history. Need at least {self.sequence_length} games.")
        
        # Scale input
        X = recent_games[feature_cols].values.reshape(1, self.sequence_length, -1)
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.feature_scaler.transform(X_reshaped).reshape(X.shape)
        
        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor)
            predictions = self.target_scaler.inverse_transform(
                predictions_scaled.cpu().numpy()
            )
        
        # Format predictions
        stats_dict = {
            'PTS': predictions[0, 0],
            'REB': predictions[0, 1],
            'AST': predictions[0, 2],
            'STL': predictions[0, 3],
            'BLK': predictions[0, 4],
            'MIN': predictions[0, 5]
        }
        
        # Calculate fantasy points
        stats_dict['FANTASY_PTS'] = (
            stats_dict['PTS'] +
            1.2 * stats_dict['REB'] +
            1.5 * stats_dict['AST'] +
            3.0 * stats_dict['STL'] +
            3.0 * stats_dict['BLK']
        )
        
        return stats_dict

    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            
        Returns:
            Dict: Evaluation metrics
        """
        # Prepare sequences
        X_test, y_test = self.prepare_sequences(test_data, self.sequence_length)
        
        # Scale data
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_test_tensor)
            predictions = self.target_scaler.inverse_transform(
                predictions_scaled.cpu().numpy()
            )
        
        # Calculate metrics
        metrics = {}
        stat_names = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN']
        
        for i, stat in enumerate(stat_names):
            mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
            
            metrics[f'{stat}_MAE'] = mae
            metrics[f'{stat}_RMSE'] = rmse
        
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