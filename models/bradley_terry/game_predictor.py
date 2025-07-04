import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BradleyTerryPredictor:
    """
    Implementation of Bradley-Terry model for predicting game outcomes.
    
    The model estimates team strengths and uses them to predict win probabilities.
    It can incorporate player statistics and other features into the team strength estimation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the predictor with configuration settings.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.learning_rate = self.config['models']['bradley_terry']['learning_rate']
        self.max_iterations = self.config['models']['bradley_terry']['max_iterations']
        self.convergence_threshold = self.config['models']['bradley_terry']['convergence_threshold']
        self.regularization = self.config['models']['bradley_terry']['regularization']
        
        self.team_strengths = {}
        self.feature_scaler = StandardScaler()
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
    def _prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Prepare and scale features for the model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_cols (List[str]): Feature column names
            
        Returns:
            np.ndarray: Scaled features
        """
        return self.feature_scaler.fit_transform(df[feature_cols])
        
    def _negative_log_likelihood(self, 
                               strengths: np.ndarray,
                               home_indices: np.ndarray,
                               away_indices: np.ndarray,
                               outcomes: np.ndarray,
                               features: Optional[np.ndarray] = None) -> float:
        """
        Calculate negative log likelihood of the Bradley-Terry model.
        
        Args:
            strengths (np.ndarray): Team strength parameters
            home_indices (np.ndarray): Indices of home teams
            away_indices (np.ndarray): Indices of away teams
            outcomes (np.ndarray): Game outcomes (1 for home win, 0 for away win)
            features (np.ndarray, optional): Additional features to modify team strengths
            
        Returns:
            float: Negative log likelihood
        """
        # Get base team strengths
        home_strengths = strengths[home_indices]
        away_strengths = strengths[away_indices]
        
        # Modify strengths with features if provided
        if features is not None:
            feature_weights = strengths[len(self.team_strengths):]
            home_strengths += np.dot(features, feature_weights)
            away_strengths += np.dot(features, feature_weights)
            
        # Calculate win probabilities
        prob_home_wins = 1 / (1 + np.exp(-(home_strengths - away_strengths)))
        
        # Calculate log likelihood
        log_likelihood = np.sum(
            outcomes * np.log(prob_home_wins) + 
            (1 - outcomes) * np.log(1 - prob_home_wins)
        )
        
        # Add regularization term
        regularization = self.regularization * np.sum(strengths ** 2)
        
        return -log_likelihood + regularization
        
    def fit(self, 
            df: pd.DataFrame,
            home_team_col: str,
            away_team_col: str,
            outcome_col: str,
            feature_cols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Fit the Bradley-Terry model to game data.
        
        Args:
            df (pd.DataFrame): Game data
            home_team_col (str): Column name for home team
            away_team_col (str): Column name for away team
            outcome_col (str): Column name for game outcome
            feature_cols (List[str], optional): Additional feature columns
            
        Returns:
            Dict[str, float]: Estimated team strengths
        """
        # Create team ID mapping
        unique_teams = pd.unique(df[[home_team_col, away_team_col]].values.ravel())
        self.team_strengths = {team: idx for idx, team in enumerate(unique_teams)}
        
        # Convert teams to indices
        home_indices = df[home_team_col].map(self.team_strengths).values
        away_indices = df[away_team_col].map(self.team_strengths).values
        outcomes = df[outcome_col].values
        
        # Prepare features if provided
        features = None
        if feature_cols is not None:
            features = self._prepare_features(df, feature_cols)
            
        # Initialize parameters
        n_params = len(self.team_strengths)
        if features is not None:
            n_params += features.shape[1]  # Add feature weights
            
        initial_strengths = np.zeros(n_params)
        
        # Optimize parameters
        result = minimize(
            self._negative_log_likelihood,
            initial_strengths,
            args=(home_indices, away_indices, outcomes, features),
            method='BFGS',
            options={
                'maxiter': self.max_iterations,
                'gtol': self.convergence_threshold
            }
        )
        
        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")
            
        # Extract and store team strengths
        final_strengths = result.x[:len(self.team_strengths)]
        strength_dict = {
            team: float(final_strengths[idx])
            for team, idx in self.team_strengths.items()
        }
        
        # Store feature weights if used
        if features is not None:
            self.feature_weights = result.x[len(self.team_strengths):]
            
        return strength_dict
        
    def predict_proba(self,
                     home_team: str,
                     away_team: str,
                     features: Optional[np.ndarray] = None) -> float:
        """
        Predict win probability for a game.
        
        Args:
            home_team (str): Home team identifier
            away_team (str): Away team identifier
            features (np.ndarray, optional): Additional features
            
        Returns:
            float: Probability of home team winning
        """
        if home_team not in self.team_strengths or away_team not in self.team_strengths:
            raise ValueError("Unknown team(s) provided")
            
        home_strength = self.team_strengths[home_team]
        away_strength = self.team_strengths[away_team]
        
        # Modify strengths with features if provided
        if features is not None and hasattr(self, 'feature_weights'):
            scaled_features = self.feature_scaler.transform(features.reshape(1, -1))
            home_strength += np.dot(scaled_features, self.feature_weights)
            away_strength += np.dot(scaled_features, self.feature_weights)
            
        return 1 / (1 + np.exp(-(home_strength - away_strength)))
        
    def get_team_rankings(self) -> pd.DataFrame:
        """
        Get current team rankings based on estimated strengths.
        
        Returns:
            pd.DataFrame: Team rankings with strengths
        """
        rankings = pd.DataFrame([
            {'team': team, 'strength': strength}
            for team, strength in self.team_strengths.items()
        ])
        
        return rankings.sort_values('strength', ascending=False).reset_index(drop=True)
        
    def save_model(self, path: str):
        """
        Save the model parameters.
        
        Args:
            path (str): Path to save the model
        """
        model_state = {
            'team_strengths': self.team_strengths,
            'feature_scaler': self.feature_scaler
        }
        
        if hasattr(self, 'feature_weights'):
            model_state['feature_weights'] = self.feature_weights
            
        np.save(path, model_state)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """
        Load model parameters.
        
        Args:
            path (str): Path to the saved model
        """
        model_state = np.load(path, allow_pickle=True).item()
        
        self.team_strengths = model_state['team_strengths']
        self.feature_scaler = model_state['feature_scaler']
        
        if 'feature_weights' in model_state:
            self.feature_weights = model_state['feature_weights']
            
        self.logger.info(f"Model loaded from {path}") 