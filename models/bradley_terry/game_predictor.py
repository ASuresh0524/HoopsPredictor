import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    brier_score_loss,
    accuracy_score
)

class BradleyTerryPredictor:
    def __init__(self, config_path="config.yaml"):
        """Initialize the Bradley-Terry predictor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model parameters
        self.learning_rate = self.config['models']['game_predictor']['learning_rate']
        self.batch_size = self.config['models']['game_predictor']['batch_size']
        self.epochs = self.config['models']['game_predictor']['epochs']
        self.early_stopping_patience = self.config['models']['game_predictor']['early_stopping_patience']
        
        # Initialize model parameters
        self.team_ratings = {}
        self.scaler = StandardScaler()
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_ratings(self, teams: List[str]):
        """Initialize team ratings."""
        for team in teams:
            if team not in self.team_ratings:
                self.team_ratings[team] = np.random.normal(0, 0.1)

    def _calculate_win_probability(self, home_rating: float, away_rating: float) -> float:
        """Calculate win probability using Bradley-Terry model."""
        return 1 / (1 + np.exp(-(home_rating - away_rating)))

    def train(
        self,
        game_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the Bradley-Terry model on historical game data.
        
        Args:
            game_data (pd.DataFrame): Game data with features
            save_path (str, optional): Path to save the trained model
            
        Returns:
            Dict: Training history
        """
        # Initialize team ratings
        teams = pd.concat([
            game_data['HOME_TEAM'],
            game_data['AWAY_TEAM']
        ]).unique()
        self._initialize_ratings(teams)
        
        # Scale features
        feature_cols = [col for col in game_data.columns if 'DIFF' in col]
        if feature_cols:
            self.scaler.fit(game_data[feature_cols])
        
        # Training loop
        history = {
            'loss': [],
            'val_loss': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            # Update ratings
            for _, game in game_data.iterrows():
                home_team = game['HOME_TEAM']
                away_team = game['AWAY_TEAM']
                
                # Get current ratings
                home_rating = self.team_ratings[home_team]
                away_rating = self.team_ratings[away_team]
                
                # Calculate predicted probability
                pred_prob = self._calculate_win_probability(home_rating, away_rating)
                
                # Get actual outcome
                actual_outcome = game['HOME_WIN']
                
                # Calculate gradient and update ratings
                error = actual_outcome - pred_prob
                gradient = error * pred_prob * (1 - pred_prob)
                
                self.team_ratings[home_team] += self.learning_rate * gradient
                self.team_ratings[away_team] -= self.learning_rate * gradient
                
                epoch_loss += -actual_outcome * np.log(pred_prob) - (1 - actual_outcome) * np.log(1 - pred_prob)
            
            avg_loss = epoch_loss / len(game_data)
            history['loss'].append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    self.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return history

    def predict(self, game_features: pd.DataFrame) -> Dict:
        """
        Predict game outcomes.
        
        Args:
            game_features (pd.DataFrame): Game features
            
        Returns:
            Dict: Predictions including win probability, spread, and total
        """
        predictions = {}
        
        for _, game in game_features.iterrows():
            home_team = game['HOME_TEAM']
            away_team = game['AWAY_TEAM']
            
            # Get team ratings
            home_rating = self.team_ratings.get(home_team, 0)
            away_rating = self.team_ratings.get(away_team, 0)
            
            # Calculate base probability from ratings
            base_prob = self._calculate_win_probability(home_rating, away_rating)
            
            # Adjust probability using features
            feature_cols = [col for col in game_features.columns if 'DIFF' in col]
            if feature_cols:
                features = game[feature_cols].values.reshape(1, -1)
                scaled_features = self.scaler.transform(features)
                feature_adjustment = np.mean(scaled_features) * 0.1  # Small adjustment based on features
                
                win_probability = np.clip(base_prob + feature_adjustment, 0.01, 0.99)
            else:
                win_probability = base_prob
            
            # Calculate spread and total
            rating_diff = home_rating - away_rating
            predicted_spread = rating_diff * 5  # Convert rating difference to points
            predicted_total = 220 + rating_diff  # Base total adjusted by rating difference
            
            predictions = {
                'win_probability': win_probability,
                'spread': predicted_spread,
                'total': predicted_total
            }
        
        return predictions

    def evaluate(self, game_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            game_data (pd.DataFrame): Game data with actual outcomes
            
        Returns:
            Dict: Evaluation metrics
        """
        y_true = game_data['HOME_WIN'].values
        y_pred_proba = []
        spreads = []
        totals = []
        
        for _, game in game_data.iterrows():
            predictions = self.predict(pd.DataFrame([game]))
            y_pred_proba.append(predictions['win_probability'])
            spreads.append(predictions['spread'])
            totals.append(predictions['total'])
        
        y_pred = np.array(y_pred_proba) > 0.5
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'spread_mae': mean_absolute_error(game_data['HOME_PTS'] - game_data['AWAY_PTS'], spreads),
            'total_mae': mean_absolute_error(game_data['HOME_PTS'] + game_data['AWAY_PTS'], totals)
        }

    def save(self, path: str):
        """Save the model to disk."""
        model_state = {
            'team_ratings': self.team_ratings,
            'scaler': self.scaler
        }
        joblib.dump(model_state, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load the model from disk."""
        model_state = joblib.load(path)
        self.team_ratings = model_state['team_ratings']
        self.scaler = model_state['scaler']
        self.logger.info(f"Model loaded from {path}") 