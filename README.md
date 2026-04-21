# AI-Mule-Account-Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Repo](https://img.shields.io/badge/GitHub-AI--Mule--Account--Detection-black)](https://github.com/SaiSugeet/AI-Mule-Account-Detection.git)

## Overview

**AI-Mule-Account-Detection** is a machine learning based fraud detection project that identifies suspicious UPI accounts that may behave like mule accounts. Mule accounts are commonly used to receive, split, and quickly forward illicit funds, making them difficult to detect with simple rule-based checks.

This project simulates UPI transaction behavior, engineers account-level and graph-based risk signals, trains a RandomForest classifier, and provides a Streamlit dashboard for interactive transaction scoring. The dashboard allows a user to enter a sender, receiver, and transaction amount, then returns a risk score, decision recommendation, top contributing signals, and an optional network view of suspicious account relationships.

The project is designed as an end-to-end fraud analytics prototype covering data simulation, feature engineering, model training, scoring, and demo-ready visualization.

## Features

- Simulates UPI accounts, VPAs, devices, transaction flows, and mule-like behavior.
- Injects realistic mule patterns such as many-to-one inbound transfers, fast forwarding, circular flows, bursts, and device reuse.
- Builds receiver-account risk features over 24-hour and 7-day windows.
- Uses graph analytics with NetworkX, including in-degree, out-degree, and PageRank-style centrality.
- Trains a RandomForest model for account risk classification.
- Provides a reusable scoring module for transaction-level inference.
- Returns decision bands: `ALLOW`, `OTP`, or `HOLD`.
- Displays top risk reasons based on deviation from feature medians.
- Includes an interactive Streamlit dashboard with theme controls, presets, threshold sliders, model feature inspection, and network visualization.

## Architecture / How It Works

The project follows a practical fraud-detection workflow:

1. **Data Simulation**
   - `simulate_data.ipynb` creates synthetic accounts, labels, devices, and UPI-style transactions.
   - Mule behavior is injected by generating abnormal transaction patterns such as rapid receive-and-forward activity and clustered account relationships.

2. **Feature Engineering**
   - `feature_engineering.ipynb` aggregates transaction history into account-level features.
   - Features include inbound/outbound transaction volume, amount totals, unique counterparties, forwarding ratio, average forwarding delay, burstiness, and graph centrality.

3. **Model Training**
   - `train_model.ipynb` trains a RandomForest classifier using engineered account features.
   - Model artifacts are saved under `model/`, including the trained model, feature importances, and decision policy thresholds.

4. **Scoring Layer**
   - `Scoring.py` loads the trained model and computes fresh receiver-account features from transaction history.
   - It returns a fraud probability, decision recommendation, reason signals, and optional ego-network graph metadata.

5. **Dashboard**
   - `app.py` runs a Streamlit dashboard for interactive scoring.
   - Users can test transactions, tune decision thresholds, inspect account features, and view network evidence.

## Tech Stack

### Machine Learning / AI

- Python
- scikit-learn
- RandomForestClassifier
- joblib
- pandas
- NumPy

### Backend / APIs

- Python scoring module
- Streamlit runtime
- Cached transaction scoring with `functools.lru_cache`

### Cloud / Deployment

- Currently configured for local Streamlit execution.
- Can be deployed to Streamlit Community Cloud, Render, Hugging Face Spaces, or any Python-compatible cloud service.

### Tools / Libraries

- NetworkX for graph-based fraud signals
- PyVis for network visualization
- Jupyter Notebook for simulation, feature engineering, and model training
- CSV and JSON artifacts for data/model interoperability

## Performance & Metrics

| Metric | Value |
|---|---:|
| Simulated accounts | 5,000 |
| Simulated mule accounts | 200 |
| Mule account ratio | 4% |
| Simulated transactions | 27,048 |
| Engineered account feature rows | 4,387 |
| Model input features | 20 |
| Model type | RandomForestClassifier |
| Local scoring latency | ~189 ms per transaction |
| Decision bands | `ALLOW`, `OTP`, `HOLD` |

### Model Performance

The current notebook run reports **100% validation accuracy**, but the validation split contains only non-mule samples after feature generation. Because of that, ROC-AUC and PR-AUC are unavailable for the saved run, and the reported accuracy should be treated as a demo baseline rather than a production fraud metric.

Current observed training output:

- Accuracy: `1.0000` on 878 validation records
- ROC-AUC: unavailable because the validation set has one class
- PR-AUC: unavailable because the validation set has one class
- Selected threshold: `0.0`
- Top model signals:
  - `avg_forward_delay_seconds`
  - `in_degree_7d`
  - `inbound_count_7d`
  - `inbound_unique_senders_7d`
  - `pagerank_7d`

For a production-grade evaluation, the engineered feature table should preserve mule labels for both normal and mule accounts, then the model should be re-evaluated using stratified train/test splits with precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix reporting.

## Getting Started

### Prerequisites

Install Python 3.12 or later. A virtual environment is recommended.

Required Python packages:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- networkx
- pyvis
- notebook

> Note: The saved model artifact was created with a newer scikit-learn build than the local environment used during verification. If you see a model persistence warning, install the same scikit-learn version used during training or retrain the model locally.

### Installation

Clone the repository:

```bash
git clone https://github.com/SaiSugeet/AI-Mule-Account-Detection.git
cd AI-Mule-Account-Detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib networkx pyvis notebook
```

### Run the Project

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### Optional: Regenerate Data and Model

Run the notebooks in this order:

```text
simulate_data.ipynb
feature_engineering.ipynb
train_model.ipynb
```

This will regenerate simulated transaction data, account-level features, and model artifacts.

## Deployment

The project is currently set up for local Streamlit execution. A live deployment can be added after the README is published.

Recommended deployment options:

- **Streamlit Community Cloud** for the fastest portfolio deployment.
- **Render** for a general Python web-service deployment.
- **Hugging Face Spaces** for ML demo hosting.

For Streamlit Community Cloud:

1. Push the project to GitHub.
2. Add a `requirements.txt` file with the required dependencies.
3. Create a new Streamlit app from the GitHub repository.
4. Set the entry point as:

```text
app.py
```

## Project Structure

```text
AI-Mule-Account-Detection/
|-- app.py                         # Streamlit dashboard for transaction scoring
|-- Scoring.py                     # Model loading, feature computation, and scoring logic
|-- simulate_data.ipynb            # Synthetic UPI account and transaction simulation
|-- feature_engineering.ipynb      # Account-level and graph-based feature engineering
|-- train_model.ipynb              # RandomForest training and artifact generation
|-- debug_predict.py               # Debug script for model prediction inputs/outputs
|-- diagnostic_model.py            # Model inspection utility
|-- diagnose_model_sample.py       # Feature importance and sample diagnostics
|-- inspect_model.py               # Model artifact inspection script
|-- data/
|   `-- features_accounts.csv      # Engineered account-level feature table
|-- raw_data/
|   |-- accounts_simulated.csv     # Simulated account metadata
|   |-- account_labels.csv         # Account labels for normal/mule accounts
|   |-- simulated_transactions.csv # Full simulated transaction dataset
|   `-- simulated_transactions_sample.csv
|-- model/
|   |-- model.pkl                  # Trained RandomForest model
|   |-- rf_baseline.pkl            # Baseline model artifact
|   |-- policy.json                # Decision threshold policy
|   `-- feature_importances.csv    # Model feature importance output
|-- graphs/
|   `-- ego_example_A002277.json   # Example account network graph
|-- lib/                           # Local frontend visualization assets
`-- LICENSE                        # MIT license
```

## Use Cases

- UPI fraud and mule-account detection prototype.
- Bank or fintech fraud analytics proof of concept.
- Graph-based financial crime detection demonstration.
- ML portfolio project for fraud detection, feature engineering, and Streamlit deployment.
- Educational project for understanding transaction-risk scoring pipelines.

## Future Improvements

- Fix the feature-label join so mule labels are preserved in the engineered feature table.
- Add a reproducible `requirements.txt` file.
- Add model evaluation reports with precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix.
- Add batch transaction scoring for uploaded CSV files.
- Add SHAP-based model explanations for more interpretable risk reasons.
- Add real-time streaming support using Kafka or a message queue.
- Add database storage for accounts, transactions, predictions, and audit history.
- Add authentication and role-based access for analyst workflows.
- Deploy the dashboard publicly using Streamlit Community Cloud or Render.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Make your changes with clear commit messages.
4. Open a pull request with a short explanation of the improvement.

## License

This project is licensed under the [MIT License](LICENSE).
