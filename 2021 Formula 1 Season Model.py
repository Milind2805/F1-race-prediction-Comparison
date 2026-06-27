import fastf1
import fastf1.plotting
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.disable(logging.CRITICAL) 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns 
# Final race results
# 2021 race calendar
races_2021 = [
    'Bahrain Grand Prix', 'Emilia Romagna Grand Prix',
    'Portuguese Grand Prix', 'Spanish Grand Prix',
    'Monaco Grand Prix', 'Azerbaijan Grand Prix',
    'French Grand Prix', 'Styrian Grand Prix',
    'Austrian Grand Prix', 'British Grand Prix',
    'Hungarian Grand Prix', 'Belgian Grand Prix',
    'Dutch Grand Prix', 'Italian Grand Prix',
    'Russian Grand Prix', 'Turkish Grand Prix',
    'United States Grand Prix', 'Mexico City Grand Prix',
    'São Paulo Grand Prix', 'Qatar Grand Prix',
    'Saudi Arabian Grand Prix', 'Abu Dhabi Grand Prix'
]

all_results = []

for race_name in races_2021:
    try:
        session = fastf1.get_session(2021, race_name, 'R')
        session.load()
        
        results = session.results[['Abbreviation', 'TeamName', 
                                    'Position', 'GridPosition', 'Points']]
        results['Race'] = race_name
        results['Year'] = 2021
        
        all_results.append(results)
        print(f"Loaded: {race_name}")
        
    except Exception as e:
        print(f"Failed: {race_name} — {e}")

# Combine all races
full_season = pd.concat(all_results, ignore_index=True)
print(full_season.shape)
print(full_season.head(40))

# # Save to CSV so you don't have to reload every time
full_season.to_csv('f1_2021_season.csv', index=False)
full_season = pd.read_csv('f1_2021_season.csv')  
print("Race data loaded:", full_season.shape)

all_quali = []

for race_name in races_2021:
    try:
        session = fastf1.get_session(2021, race_name, 'Q')
        session.load()
        
        quali = session.results[['Abbreviation', 'Position']]
        quali = quali.rename(columns={'Position': 'QualiPosition'})
        quali['Race'] = race_name
        quali['Year'] = 2021
        
        all_quali.append(quali)
        print(f"Loaded Quali: {race_name}")
        
    except Exception as e:
        print(f"Failed: {race_name} — {e}")

quali_season = pd.concat(all_quali, ignore_index=True)

# Merge race results with qualifying
full_dataset = pd.merge(full_season, quali_season, 
                         on=['Abbreviation', 'Race', 'Year'])

print(full_dataset.shape)
print(full_dataset.head())

# # Save
full_dataset.to_csv('f1_2021_full.csv', index=False)
full_dataset = pd.read_csv('f1_2021_full.csv')

