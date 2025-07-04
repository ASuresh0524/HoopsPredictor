# HoopsPredictor

A comprehensive NBA game prediction system that combines team statistics, player performance, and historical data to predict game outcomes and player statistics.

## Features

- Team-level predictions using Bradley-Terry model
- Player statistics forecasting using LSTM networks
- Advanced feature engineering including:
  - Rolling statistics
  - Team momentum indicators
  - Head-to-head matchup analysis
  - Rest day impact
- Comprehensive evaluation metrics
- Configurable parameters via YAML

## Project Structure

```
HoopsPredictor/
├── config.yaml           # Configuration parameters
├── requirements.txt      # Python dependencies
├── TeamData.sqlite      # Team-level statistics database
├── dataset.sqlite       # Player-level statistics database
├── models/
│   ├── bradley_terry/   # Team prediction models
│   └── time_series/     # Player stats prediction models
├── utils/
│   ├── data_loader.py   # Database handling
│   └── feature_engineering.py # Feature creation
└── pipelines/
    └── prediction_pipeline.py # Main prediction pipeline
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HoopsPredictor.git
cd HoopsPredictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure parameters in `config.yaml`
2. Run predictions:

```python
from pipelines.prediction_pipeline import PredictionPipeline

# Initialize pipeline
pipeline = PredictionPipeline()

# Get predictions for upcoming games
predictions = pipeline.predict_upcoming_games()

# Get player stat predictions
player_predictions = pipeline.predict_player_stats(player_id="1234")
```

## Data Sources

The project uses two SQLite databases:

1. `TeamData.sqlite`: Contains daily team statistics
   - Tables named by date (e.g., "2024-04-29")
   - Includes team performance metrics, shooting percentages, etc.

2. `dataset.sqlite`: Contains player-level statistics
   - Main table: "dataset_2012-24"
   - Includes individual player performance data

## Model Details

### Bradley-Terry Model
- Used for team-vs-team outcome prediction
- Incorporates team strength ratings
- Accounts for home court advantage

### LSTM Network
- Predicts player statistics
- Uses sequence of past performances
- Handles variable-length input sequences

## Configuration

Key parameters in `config.yaml`:

```yaml
data:
  train_start_date: "2012-01-01"
  train_end_date: "2023-06-30"
  val_start_date: "2023-07-01"
  val_end_date: "2023-12-31"
  test_start_date: "2024-01-01"
  test_end_date: "2024-04-29"

feature_engineering:
  rolling_windows: [5, 10, 20]
  momentum_windows: [3, 5, 10]
  head2head_lookback_days: 365
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NBA Stats API for data access
- PyTorch team for the deep learning framework
- Contributors and maintainers 