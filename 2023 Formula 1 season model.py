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
race = fastf1.get_session(2023, 'British Grand Prix', 'R')
race.load()
laps = race.laps
#print(laps.shape)        
#print(laps.columns.tolist())  
#print(laps.head())      

hamilton_laps = race.laps.pick_drivers('HAM')
verstappen_laps = race.laps.pick_drivers('VER')


top_drivers = race.laps.pick_drivers(['HAM', 'VER', 'LEC'])

laps = race.laps.copy()


laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
laps['Sector1Seconds'] = laps['Sector1Time'].dt.total_seconds()
laps['Sector2Seconds'] = laps['Sector2Time'].dt.total_seconds()
laps['Sector3Seconds'] = laps['Sector3Time'].dt.total_seconds()

laps = laps.dropna(subset=['LapTimeSeconds'])


# Final race results
results = race.results
#print(results[['Abbreviation', 'TeamName', 'Position', 'GridPosition', 'Points', 'Status']])
weather = race.weather_data
#print(weather.columns.tolist())
#print(weather.head(52))
# Get fastest lap for Hamilton
hamilton_fastest = race.laps.pick_drivers('HAM').pick_fastest()

# Get telemetry for that lap
telemetry = hamilton_fastest.get_telemetry()
#print(telemetry.columns.tolist())
#print(telemetry['Speed'].head(4))

# 2023 race calendar
races_2023 = [
    'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 
    'Australian Grand Prix', 'Azerbaijan Grand Prix',
    'Miami Grand Prix', 'Monaco Grand Prix',
    'Spanish Grand Prix', 'Canadian Grand Prix',
    'Austrian Grand Prix', 'British Grand Prix',
    'Hungarian Grand Prix', 'Belgian Grand Prix',
    'Dutch Grand Prix', 'Italian Grand Prix',
    'Singapore Grand Prix', 'Japanese Grand Prix',
    'Qatar Grand Prix', 'United States Grand Prix',
    'Mexico City Grand Prix', 'São Paulo Grand Prix',
    'Las Vegas Grand Prix', 'Abu Dhabi Grand Prix']


all_results = []

for race_name in races_2023:
    try:
        session = fastf1.get_session(2023, race_name, 'R')
        session.load()
        
        results = session.results[['Abbreviation', 'TeamName', 
                                    'Position', 'GridPosition', 'Points']]
        results['Race'] = race_name
        results['Year'] = 2023
        
        all_results.append(results)
        print(f"Loaded: {race_name}")
        
    except Exception as e:
        print(f"Failed: {race_name} — {e}")

# Combine all races
full_season = pd.concat(all_results, ignore_index=True)
print(full_season.shape)
print(full_season.head(40))

# # Save to CSV so you don't have to reload every time
# full_season.to_csv('f1_2023_season.csv', index=False)
full_season = pd.read_csv('f1_2023_season.csv')
print("Race data loaded:", full_season.shape)

all_quali = []

for race_name in races_2023:
    try:
        session = fastf1.get_session(2023, race_name, 'Q')
        session.load()
        
        quali = session.results[['Abbreviation', 'Position']]
        quali = quali.rename(columns={'Position': 'QualiPosition'})
        quali['Race'] = race_name
        quali['Year'] = 2023
        
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

# Save
full_dataset.to_csv('f1_2023_full.csv', index=False)
full_dataset = pd.read_csv('f1_2023_full.csv')

# # 1 = Winner, 0 = Did not win
# full_dataset['Winner'] = (full_dataset['Position'] == 1).astype(int)

# print(full_dataset['Winner'].value_counts())
# # Should show: 0 → 418, 1 → 22 (one winner per race)
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("f1_2023_full.csv")
df['Winner'] = (df['Position'] == 1).astype(int)
# print(df.head(3),df.shape)
le=LabelEncoder()
df['Driver']=le.fit_transform(df['Abbreviation'])
df['Team']=le.fit_transform(df['TeamName'])
df['Circuit']=le.fit_transform(df['Race'])
df = df.dropna(subset=['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit'])
x = df[['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit']]
y = df['Winner']
#print(x.isnull().sum())  # Check which columns have NaN

print("Class distribution:", y.value_counts())
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_resampled, y_resampled = sm.fit_resample(x, y)
print("After SMOTE:", y_resampled.value_counts())
x_resampled_train, x_resampled_test, y_resampled_train, y_resampled_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)
ss=StandardScaler()
x_resampled_train = ss.fit_transform(x_resampled_train)
x_resampled_test = ss.transform(x_resampled_test)
# print("Training set size:", x_resampled_train.shape)
# print("Test set size:", x_resampled_test.shape)
xgb=XGBClassifier()
xgb.fit(x_resampled_train,y_resampled_train)
print("XGB TRAIN:",xgb.score(x_resampled_train,y_resampled_train)*100)
print("XGB TEST:",xgb.score(x_resampled_test,y_resampled_test)*100)
y_pred = xgb.predict(x_resampled_test)
# cm = confusion_matrix(y_resampled_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - XGBoost")
#plt.show()
#from xgboost import plot_importance
#plot_importance(xgb, ylabel='Features',feature_names=['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit'])
#plt.title('Feature Importance')
#plt.show()
feature_names = ['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit']
importance_scores = xgb.feature_importances_

# sns.barplot(x=importance_scores, y=feature_names)
# plt.title('Feature Importance')
# plt.xlabel('Importance Score')
# plt.show()
# top_drivers = df.groupby('Abbreviation')['Points'].sum().nlargest(5).index

# plt.figure(figsize=(12, 5))
# for driver in top_drivers:
#     driver_data = df[df['Abbreviation'] == driver]
#     cumulative_points = driver_data['Points'].cumsum()
#     plt.plot(range(1, len(cumulative_points)+1), 
#              cumulative_points, marker='o', label=driver)

# plt.title('Championship Points Progression - 2023')
# plt.xlabel('Race Number')
# plt.ylabel('Cumulative Points')
# plt.legend()
# plt.grid(True)
# plt.show()
# winners = df[df['Position'] == 1]['Abbreviation'].value_counts()
# sns.barplot(x=winners.values, y=winners.index, palette='Reds_r')
# #plt.figure(figsize=(10, 6))
# plt.title('Race Wins by Driver - 2023')
# plt.xlabel('Number of Wins')
# plt.ylabel('Driver')
# plt.show()
# team_wins = df[df['Position'] == 1]['TeamName'].value_counts()
# sns.barplot(x=team_wins.values, y=team_wins.index, palette='Blues_r')
# plt.title('Race Wins by Constructor - 2023')
# plt.xlabel('Number of Wins')
# plt.ylabel('Constructor')
# plt.show()
# constructor_points = df.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
# sns.barplot(x=constructor_points.values, y=constructor_points.index, palette='viridis')
# plt.title('Total Points by Constructor - 2023')
# plt.xlabel('Total Points')
# plt.ylabel('Constructor')
# plt.show()
df = pd.read_csv('f1_2023_full.csv')
print(df['Race'].unique())  # Check which races are present
print(df.shape)             # Should be 440 rows