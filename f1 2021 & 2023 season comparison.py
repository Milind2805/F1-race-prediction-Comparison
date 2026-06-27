import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
df_2023 = pd.read_csv('f1_2023_full.csv')
df_2021 = pd.read_csv('f1_2021_full.csv')
# For visualizations — use raw unprocessed data
df_2023_viz = pd.read_csv('f1_2023_full.csv')
df_2021_viz = pd.read_csv('f1_2021_full.csv')

df_2023_viz['Season'] = 2023
df_2021_viz['Season'] = 2021
df_2023_viz['Winner'] = (df_2023_viz['Position'] == 1).astype(int)
df_2021_viz['Winner'] = (df_2021_viz['Position'] == 1).astype(int)

# For ML model — use cleaned data (what you already have)
# For ML model — use original column names
df_2023 = df_2023_viz.dropna(subset=['Abbreviation', 'TeamName', 'GridPosition', 'QualiPosition', 'Race'])
df_2021 = df_2021_viz.dropna(subset=['Abbreviation', 'TeamName', 'GridPosition', 'QualiPosition', 'Race'])

df_2023['Season'] = 2023
df_2021['Season'] = 2021

print("2023 Shape:", df_2023.shape)
print("2021 Shape:", df_2021.shape)
df_2021['Winner'] = (df_2021['Position'] == 1).astype(int)
df_2023['Winner'] = (df_2023['Position'] == 1).astype(int)
print("2021 Columns:", df_2021.columns)
print("2023 Columns:", df_2023.columns)
le=LabelEncoder()
df_2021['Driver']=le.fit_transform(df_2021['Abbreviation'])
df_2021['Team']=le.fit_transform(df_2021['TeamName'])
df_2021['Circuit']=le.fit_transform(df_2021['Race'])
df_2023['Driver']=le.fit_transform(df_2023['Abbreviation'])
df_2023['Team']=le.fit_transform(df_2023['TeamName'])  
df_2021['Team']=le.fit_transform(df_2021['TeamName']) 
df_2023['Circuit']=le.fit_transform(df_2023['Race']) 
df_2021_clean = df_2021.dropna(subset=['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit'])
df_2023_clean = df_2023.dropna(subset=['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit'])
x1 = df_2021_clean[['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit']]
y1 = df_2021_clean['Winner']
x2= df_2023_clean[['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit']]
y2 = df_2023_clean['Winner']
# print(x1.isnull().sum())
# print(x2.isnull().sum())
print("Class distribution:", y1.value_counts())
print("Class distribution:", y2.value_counts())
sm = SMOTE(random_state=42)
x_resampled1, y_resampled1 = sm.fit_resample(x1, y1)
x_resampled2, y_resampled2 = sm.fit_resample(x2, y2)

# print("After SMOTE:", y_resampled1.value_counts())
# print("After SMOTE:", y_resampled2.value_counts())
x_resampled1_train, x_resampled1_test, y_resampled1_train, y_resampled1_test = train_test_split(x_resampled1, y_resampled1, test_size=0.2, random_state=42)
x_resampled2_train, x_resampled2_test, y_resampled2_train, y_resampled2_test = train_test_split(x_resampled2, y_resampled2, test_size=0.2, random_state=42)
ss=StandardScaler()
x_resampled1_train = ss.fit_transform(x_resampled1_train)
x_resampled1_test = ss.transform(x_resampled1_test)
x_resampled2_train = ss.fit_transform(x_resampled2_train)
x_resampled2_test = ss.transform(x_resampled2_test)
# print("Training set size:", x_resampled1_train.shape)
# print("Test set size:", x_resampled1_test.shape)
# print("Training set size:", x_resampled2_train.shape)
# print("Test set size:", x_resampled2_test.shape)
xgb=XGBClassifier()
xgb.fit(x_resampled1_train,y_resampled1_train)
print("XGB TRAIN 2021:",xgb.score(x_resampled1_train,y_resampled1_train)*100)
print("XGB TEST 2021:",xgb.score(x_resampled1_test,y_resampled1_test)*100)
y_pred1 = xgb.predict(x_resampled1_test)  
xgb2=XGBClassifier()
xgb2.fit(x_resampled2_train,y_resampled2_train)
print("XGB TRAIN 2023:",xgb2.score(x_resampled2_train,y_resampled2_train)*100)
print("XGB TEST 2023:",xgb2.score(x_resampled2_test,y_resampled2_test)*100)
y_pred2 = xgb2.predict(x_resampled2_test)
# top_drivers = df_2021.groupby('Abbreviation')['Points'].sum().nlargest(5).index

# plt.figure(figsize=(12, 5))
# for driver in top_drivers:
#     driver_data = df_2021[df_2021['Abbreviation'] == driver]
#     cumulative_points = driver_data['Points'].cumsum()
#     plt.plot(range(1, len(cumulative_points)+1), 
#              cumulative_points, marker='o', label=driver)

# plt.title('Championship Points Progression - 2021')
# plt.xlabel('Race Number')
# plt.ylabel('Cumulative Points')
# plt.legend()
# plt.grid(True)
# plt.show() 
# top_drivers = df_2023.groupby('Abbreviation')['Points'].sum().nlargest(5).index

# plt.figure(figsize=(12, 5))
# for driver in top_drivers:
#     driver_data = df_2023[df_2023['Abbreviation'] == driver]
#     cumulative_points = driver_data['Points'].cumsum()
#     plt.plot(range(1, len(cumulative_points)+1), 
#              cumulative_points, marker='o', label=driver)

# plt.title('Championship Points Progression - 2023')
# plt.xlabel('Race Number')
# plt.ylabel('Cumulative Points')
# plt.legend()
# plt.grid(True)
# plt.show()
# winners = df_2021[df_2021['Position'] == 1]['Abbreviation'].value_counts()
# sns.barplot(x=winners.values, y=winners.index, palette='Reds_r')
# #plt.figure(figsize=(10, 6))
# plt.title('Race Wins by Driver - 2021')
# plt.xlabel('Number of Wins')
# plt.ylabel('Driver')
# plt.show()
# team_wins = df_2021[df_2021['Position'] == 1]['TeamName'].value_counts()
# sns.barplot(x=team_wins.values, y=team_wins.index, palette='Blues_r')
# plt.title('Race Wins by Constructor - 2021')
# plt.xlabel('Number of Wins')
# plt.ylabel('Constructor')
# plt.show()
# constructor_points = df_2021.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
# sns.barplot(x=constructor_points.values, y=constructor_points.index, palette='viridis')
# plt.title('Total Points by Constructor - 2021')
# plt.xlabel('Total Points')
# plt.ylabel('Constructor')
# plt.show()

# winners = df_2023[df_2023['Position'] == 1]['Abbreviation'].value_counts()
# sns.barplot(x=winners.values, y=winners.index, palette='Reds_r')
# #plt.figure(figsize=(10, 6))
# plt.title('Race Wins by Driver - 2023')
# plt.xlabel('Number of Wins')
# plt.ylabel('Driver')
# plt.show()
# team_wins = df_2023[df_2023['Position'] == 1]['TeamName'].value_counts()
# sns.barplot(x=team_wins.values, y=team_wins.index, palette='Blues_r')
# plt.title('Race Wins by Constructor - 2023')
# plt.xlabel('Number of Wins')
# plt.ylabel('Constructor')
# plt.show()
# constructor_points = df_2023.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
# sns.barplot(x=constructor_points.values, y=constructor_points.index, palette='viridis')
# plt.title('Total Points by Constructor - 2023')
# plt.xlabel('Total Points')
# plt.ylabel('Constructor')
# plt.show()
seasons = ['2021', '2023']
accuracies = [97.00, 99.40]

# sns.barplot(x=seasons, y=accuracies, palette=['#00D2BE', '#0600EF'])
# plt.title('XGBoost Model Accuracy — 2021 vs 2023')
# plt.ylabel('Test Accuracy (%)')
# plt.ylim(90, 100)
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# wins_2021 = df_2021[df_2021['Position']==1]['Abbreviation'].value_counts()
# wins_2023 = df_2023[df_2023['Position']==1]['Abbreviation'].value_counts()

# sns.barplot(x=wins_2021.values, y=wins_2021.index, ax=axes[0], palette='Blues_r')
# axes[0].set_title('Race Wins - 2021')

# sns.barplot(x=wins_2023.values, y=wins_2023.index, ax=axes[1], palette='Reds_r')
# axes[1].set_title('Race Wins - 2023')

# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# # 2021 top drivers
# top_2021 = ['VER', 'HAM', 'BOT', 'PER']
# for driver in top_2021:
#     data = df_2021[df_2021['Abbreviation']==driver]
#     axes[0].plot(range(1, len(data)+1), 
#                  data['Points'].cumsum(), 
#                  marker='o', label=driver)
# axes[0].set_title('Championship Battle - 2021')
# axes[0].set_xlabel('Race Number')
# axes[0].set_ylabel('Cumulative Points')
# axes[0].legend()
# axes[0].grid(True)

# # 2023 top drivers
# top_2023 = ['VER', 'PER', 'HAM', 'ALO']
# for driver in top_2023:
#     data = df_2023[df_2023['Abbreviation']==driver]
#     axes[1].plot(range(1, len(data)+1), 
#                  data['Points'].cumsum(), 
#                  marker='o', label=driver)
# axes[1].set_title('Championship Dominance - 2023')
# axes[1].set_xlabel('Race Number')
# axes[1].set_ylabel('Cumulative Points')
# axes[1].legend()
# axes[1].grid(True)

# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# for df, ax, year, leader in [(df_2021, axes[0], '2021', 'VER'), 
#                               (df_2023, axes[1], '2023', 'VER')]:
#     top5 = df.groupby('Abbreviation')['Points'].sum().nlargest(5).index
#     leader_pts = df[df['Abbreviation']==leader]['Points'].cumsum().values
    
#     for driver in top5:
#         if driver != leader:
#             driver_pts = df[df['Abbreviation']==driver]['Points'].cumsum().values
#             min_len = min(len(leader_pts), len(driver_pts))
#             gap = leader_pts[:min_len] - driver_pts[:min_len]
#             ax.plot(range(1, min_len+1), gap, marker='o', label=driver)
    
#     ax.set_title(f'Points Gap to VER - {year}')
#     ax.set_xlabel('Race Number')
#     ax.set_ylabel('Points Behind Leader')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.show()
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# # Championship progression
# top_2023 = ['VER', 'PER', 'HAM', 'ALO', 'SAI']
# df_top = df_2023_viz[df_2023_viz['Abbreviation'].isin(top_2023)].copy()
# df_top['CumulativePoints'] = df_top.groupby('Abbreviation')['Points'].cumsum()
# df_top['RaceNumber'] = df_top.groupby('Abbreviation').cumcount() + 1

# fig = px.line(df_top, x='RaceNumber', y='CumulativePoints',
#               color='Abbreviation', markers=True)
# fig.update_layout(title='Championship Points Progression - 2023',
#                   xaxis_title='Race Number',
#                   yaxis_title='Cumulative Points')
# fig.show()

# # Win distribution
# wins_2023 = df_2023_viz[df_2023_viz['Position']==1]['Abbreviation'].value_counts()
# wins_2021 = df_2021_viz[df_2021_viz['Position']==1]['Abbreviation'].value_counts()

# fig = px.bar(x=wins_2023.values, y=wins_2023.index,
#              orientation='h',
#              title='Race Wins by Driver - 2023',
#              labels={'x': 'Number of Wins', 'y': 'Driver'})
# fig.show()

# # Points by constructor
# constructor_points = df_2023_viz.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
# fig = px.bar(x=constructor_points.values, y=constructor_points.index,
#              orientation='h',
#              title='Total Points by Constructor - 2023',
#              labels={'x': 'Total Points', 'y': 'Constructor'})
# fig.show()

# # Accuracy comparison
# seasons = ['2021', '2023']
# accuracies = [97.0, 99.4]

# fig = px.bar(x=seasons, y=accuracies,
#              color=seasons,
#              color_discrete_map={'2021': '#00D2BE', '2023': '#0600EF'},
#              title='XGBoost Model Accuracy — 2021 vs 2023',
#              labels={'x': 'Season', 'y': 'Test Accuracy (%)'})
# fig.update_layout(yaxis_range=[90, 100],
#                   xaxis_title='Season',
#                   yaxis_title='Test Accuracy (%)')
# fig.update_traces(text=['97.0%', '99.4%'], textposition='outside')
# fig.show()

# Use df_2023_viz for accurate data
top_drivers_2023 = ['VER', 'PER', 'HAM', 'ALO', 'SAI', 'NOR', 'LEC', 'RUS']

df_anim = df_2023_viz[df_2023_viz['Abbreviation'].isin(top_drivers_2023)].copy()

df_anim = df_anim.sort_values(['Abbreviation', 'Race'])        # Sort by Race name first
df_anim['RaceNumber'] = df_anim.groupby('Abbreviation').cumcount() + 1  # Create RaceNumber
df_anim['CumulativePoints'] = df_anim.groupby('Abbreviation')['Points'].cumsum()
df_wide = df_anim.pivot(index='RaceNumber', 
                         columns='Abbreviation', 
                         values='CumulativePoints')
df_wide = df_wide.ffill().fillna(0)
#print(df_wide)
driver_colors = {
    # Red Bull
    'VER': '#3671C6',
    'PER': '#3671C6',

    # Mercedes
    'HAM': '#6CD3BF',
    'RUS': '#6CD3BF',

    # Ferrari
    'LEC': '#F91536',
    'SAI': '#F91536',

    # McLaren
    'NOR': '#FF8700',
    'PIA': '#FF8700',

    # Aston Martin
    'ALO': '#229971',
    'STR': '#229971',

    # Alpine
    'GAS': '#2293D1',
    'OCO': '#2293D1',

    # Williams
    'ALB': '#37BEDD',
    'SAR': '#37BEDD',

    # AlphaTauri
    'TSU': '#5E8FAA',
    'DEV': '#5E8FAA',
    'RIC': '#5E8FAA',
    'LAW': '#5E8FAA',

    # Alfa Romeo
    'BOT': '#900000',
    'ZHO': '#900000',

    # Haas
    'HUL': '#B6BABD',
    'MAG': '#B6BABD'
}

race_names = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Azerbaijan",
    "Miami",
    "Monaco",
    "Spain",
    "Canada",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Singapore",
    "Japan",
    "Qatar",
    "USA",
    "Mexico",
    "Brazil",
    "Las Vegas",
    "Abu Dhabi"
]

# df_wide.index = race_names

import bar_chart_race as bcr

# bcr.bar_chart_race(
#     df=df_wide,
#     filename='f1_2023_championship.gif',
#     orientation='h',
#     sort='desc',
#     n_bars=8,
#     cmap=[driver_colors[d] for d in df_wide.columns],
#     period_length=1500,
#     steps_per_period=30,
#     title='F1 2023 Championship Race — VER Dominance',
#     period_label={
#         'x':0.98,
#         'y':0.12,
#         'ha':'right',
#         'fontsize':22,
#         'color':'white'
#     },bar_kwargs={
#     'linewidth':1
# }
# )
# print("Animation saved as f1_2023_championship.gif")
df_2021_clean_viz = df_2021.sort_values('Position').drop_duplicates(
    subset=['Abbreviation', 'Race'], keep='first')
top_drivers_2021 = ['VER', 'HAM', 'BOT', 'PER', 'NOR', 'LEC', 'ALO', 'TSU']

df_anim_2021 = df_2021_clean_viz[df_2021_clean_viz['Abbreviation'].isin(top_drivers_2021)].copy()

df_anim_2021 = df_anim_2021.drop_duplicates(subset=['Abbreviation', 'Race']) 
race_order_2021 = [
    'Bahrain Grand Prix',
    'Emilia Romagna Grand Prix',
    'Portuguese Grand Prix',
    'Spanish Grand Prix',
    'Monaco Grand Prix',
    'Azerbaijan Grand Prix',
    'French Grand Prix',
    'Styrian Grand Prix',
    'Austrian Grand Prix',
    'British Grand Prix',
    'Hungarian Grand Prix',
    'Belgian Grand Prix',
    'Dutch Grand Prix',
    'Italian Grand Prix',
    'Russian Grand Prix',
    'Turkish Grand Prix',
    'United States Grand Prix',
    'Mexico City Grand Prix',
    'São Paulo Grand Prix',
    'Qatar Grand Prix',
    'Saudi Arabian Grand Prix',
    'Abu Dhabi Grand Prix'
]
race_order_map = {race: i+1 for i, race in enumerate(race_order_2021)}
df_anim_2021['RaceNumber'] = df_anim_2021['Race'].map(race_order_map) # ← Add this
df_anim_2021 = df_anim_2021.sort_values(
    ['Abbreviation', 'RaceNumber']).reset_index(drop=True)

df_anim_2021['CumulativePoints'] = df_anim_2021.groupby(
    'Abbreviation')['Points'].cumsum()
df_wide_2021 = df_anim_2021.pivot(index='Race', 
                         columns='Abbreviation', 
                         values='CumulativePoints')
print(df_anim_2021[df_anim_2021['Abbreviation']=='VER'][['Race', 'RaceNumber', 'Points', 'CumulativePoints']])
df_wide_2021 = df_wide_2021.reindex(race_order_2021)
df_wide_2021 = df_wide_2021.ffill().fillna(0)
driver_colors = {
    # Red Bull
    'VER': '#3671C6',
    'PER': '#3671C6',

    # Mercedes
    'HAM': '#6CD3BF',
    'RUS': '#6CD3BF',

    # Ferrari
    'LEC': '#F91536',
    'SAI': '#F91536',

    # McLaren
    'NOR': '#FF8700',
    'PIA': '#FF8700',

    # Aston Martin
    'ALO': '#229971',
    'STR': '#229971',

    # Alpine
    'GAS': '#2293D1',
    'OCO': '#2293D1',

    # Williams
    'ALB': '#37BEDD',
    'SAR': '#37BEDD',

    # AlphaTauri
    'TSU': '#5E8FAA',
    'DEV': '#5E8FAA',
    'RIC': '#5E8FAA',
    'LAW': '#5E8FAA',

    # Alfa Romeo
    'BOT': '#900000',
    'ZHO': '#900000',

    # Haas
    'HUL': '#B6BABD',
    'MAG': '#B6BABD'
}
# Define correct 2021 race order

print(df_wide_2021) 
# print(df_2021_viz[df_2021_viz['Abbreviation']=='VER'].shape)
# print(df_2021_viz[df_2021_viz['Abbreviation']=='VER'][['Race', 'Points', 'QualiPosition']].head(10))
# # Should be 22 rows — one per race
# If more than 22, duplicates exist # Verify order is correct
# bcr.bar_chart_race(
#     df=df_wide_2021,
#     filename='f1_2021_championship.gif',
#     orientation='h',
#     sort='desc',
#     n_bars=8,
#     cmap=[driver_colors[d] for d in df_wide_2021.columns],
#     period_length=1500,
#     steps_per_period=30,
#     title='F1 2021 Championship Race — Goated Rivalry',
#     period_label={
#         'x':0.98,
#         'y':0.12,
#         'ha':'right',
#         'fontsize':22,
#         'color':'white'
#     },bar_kwargs={
#     'linewidth':1
# }
# )
# print("Animation saved as f1_2021_championship.gif")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

feature_names = ['Driver', 'Team', 'GridPosition', 'QualiPosition', 'Circuit']

# 2021
axes[0].barh(feature_names, xgb.feature_importances_, color='steelblue')
axes[0].set_title('Feature Importance - 2021')
axes[0].set_xlabel('Importance Score')

# 2023
axes[1].barh(feature_names, xgb2.feature_importances_, color='crimson')
axes[1].set_title('Feature Importance - 2023')
axes[1].set_xlabel('Importance Score')

plt.suptitle('Feature Importance Comparison — 2021 vs 2023', fontsize=14)
plt.tight_layout()
plt.show()
