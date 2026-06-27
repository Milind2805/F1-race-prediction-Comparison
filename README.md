# F1-race-prediction-Comparison
F1 Race Winner Prediction using XGBoost and FastF1 API.  Compares 2021 vs 2023 seasons. 99.4% accuracy on 2023 data and 97% on 2021 data.
# F1 Race Winner Prediction 🏎️

A machine learning project that predicts Formula 1 race winners using 
real official F1 timing data from the FastF1 API, comparing two of the 
most contrasting seasons in recent F1 history — 2021 and 2023.

## 🏆 Results

| Season | Train Accuracy | Test Accuracy |
|--------|---------------|---------------|
| 2023   | 99.85%        | 99.40%        |
| 2021   | 99.54%        | 97.00%        |

## 🎬 Championship Race Animations

### 2021 — The Goated Rivalry
![2021 Championship](f1_2021_championship.gif)

### 2023 — The Dominant Era
![2023 Championship](f1_2023_championship.gif)

## 📌 About
Built during my first year of B.Tech Civil Engineering at MANIT Bhopal,
combining my passion for Formula 1 with Machine Learning. Uses real 
official F1 timing data via the FastF1 API to predict race winners and 
compare two contrasting seasons.

## 🛠️ Tech Stack
- Python
- XGBoost
- FastF1 API
- SMOTE (imbalanced-learn)
- Plotly
- pandas
- scikit-learn
- seaborn
- bar-chart-race

## 📊 Dataset
- Source: Official F1 timing data via FastF1 API
- 2023 Season: 22 races × 20 drivers = 440 rows
- 2021 Season: 22 races × 20 drivers = 440 rows
- Features: Driver, Constructor, Grid Position, 
  Qualifying Position, Circuit
- Target: Race Winner (1 = Winner, 0 = Did not win)
- Class imbalance handled using SMOTE

## 🔍 Key ML Insights

**Why 2023 accuracy is higher than 2021:**
In 2023 Verstappen won 19 out of 22 races making the patterns 
extremely consistent and easy to predict. In 2021 Hamilton and 
Verstappen traded wins all the way to the final lap of the final 
race making prediction significantly harder.

**Feature Importance 2023:**
- QualiPosition: 0.49 — most important by far
- Driver: 0.30 — individual driver identity matters
- Team: 0.10 — constructor performance
- GridPosition: 0.10 — starting position
- Circuit: 0.02 — barely matters in dominant era

**Feature Importance 2021:**
- More balanced across all features
- Driver identity relatively less dominant
- Circuit type mattered more in competitive season

**Overfitting Analysis:**
The 2021 model shows slightly more overfitting (2.54% gap) than 
2023 (0.45% gap) — unpredictable seasons are harder for ML models 
to generalise on, just like they're harder for F1 strategists.

## 🚀 How to Run

**Install dependencies:**
```bash
pip install fastf1 xgboost imbalanced-learn plotly 
bar-chart-race pandas scikit-learn seaborn matplotlib
```

**Run in order:**
1. `2023 Formula 1 season model.py` — collects 2023 data
2. `2021 Formula 1 Season Model.py` — collects 2021 data
3. `f1 2021 & 2023 season comparison.py` — trains models 
   and generates all visualizations

**Note:** FastF1 caches data locally after first download.
Sprint race points are excluded as only main race results 
are collected via FastF1.

## 📁 Project Structure
