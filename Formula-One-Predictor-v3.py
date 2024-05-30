# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Create and navigate to project directory
%mkdir -p gdrive/MyDrive/Colab\ Notebooks/Final_Project
%cd gdrive/MyDrive/Colab\ Notebooks/Final_Project

# Install necessary packages
!pip install -q torch_utils PyDrive selenium wandb
!apt-get update && apt-get install -y chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin

# Set up Selenium WebDriver
import sys
from selenium import webdriver

sys.path.insert(0, '/usr/lib/chromium-browser/chromedriver')
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
wd = webdriver.Chrome('chromedriver', chrome_options=chrome_options)

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tqdm
from PIL import Image
import warnings
import os
from glob import glob
from timeit import default_timer as timer
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from google.colab import files
from oauth2client.client import GoogleCredentials
import wandb
import requests
from bs4 import BeautifulSoup
from torchvision import models, transforms

warnings.filterwarnings('ignore', category=FutureWarning)

# Authenticate and initialize Google Drive and W&B
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
wandb.login()

# Define functions for data extraction and processing
def extract_races_data():
    races = {'season': [], 'round': [], 'circuit_id': [], 'lat': [], 'long': [], 'country': [], 'date': [], 'url': []}
    for year in range(2000, 2022):
        url = f'https://ergast.com/api/f1/{year}.json'
        r = requests.get(url)
        json = r.json()
        for item in json['MRData']['RaceTable']['Races']:
            races['season'].append(int(item['season']))
            races['round'].append(int(item['round']))
            races['circuit_id'].append(item['Circuit']['circuitId'])
            races['lat'].append(float(item['Circuit']['Location']['lat']))
            races['long'].append(float(item['Circuit']['Location']['long']))
            races['country'].append(item['Circuit']['Location']['country'])
            races['date'].append(item['date'])
            races['url'].append(item['url'])
    return pd.DataFrame(races)

def extract_results_data(rounds):
    results = {'season': [], 'round': [], 'circuit_id': [], 'driver': [], 'date_of_birth': [], 'nationality': [], 'constructor': [], 'grid': [], 'time': [], 'status': [], 'points': [], 'podium': [], 'url': []}
    for year_rounds in rounds:
        year, round_list = year_rounds
        for r in round_list:
            url = f'http://ergast.com/api/f1/{year}/{r}/results.json'
            r = requests.get(url)
            json = r.json()
            for item in json['MRData']['RaceTable']['Races'][0]['Results']:
                results['season'].append(int(json['MRData']['RaceTable']['Races'][0]['season']))
                results['round'].append(int(json['MRData']['RaceTable']['Races'][0]['round']))
                results['circuit_id'].append(json['MRData']['RaceTable']['Races'][0]['Circuit']['circuitId'])
                results['driver'].append(item['Driver']['driverId'])
                results['date_of_birth'].append(item['Driver']['dateOfBirth'])
                results['nationality'].append(item['Driver']['nationality'])
                results['constructor'].append(item['Constructor']['constructorId'])
                results['grid'].append(int(item['grid']))
                results['time'].append(int(item['Time']['millis']) if 'Time' in item else None)
                results['status'].append(item['status'])
                results['points'].append(int(item['points']))
                results['podium'].append(int(item['position']))
                results['url'].append(json['MRData']['RaceTable']['Races'][0]['url'])
    return pd.DataFrame(results)

def extract_driver_standings(rounds):
    driver_standings = {'season': [], 'round': [], 'driver': [], 'driver_points': [], 'driver_wins': [], 'driver_standings_pos': []}
    for year_rounds in rounds:
        year, round_list = year_rounds
        for r in round_list:
            url = f'https://ergast.com/api/f1/{year}/{r}/driverStandings.json'
            r = requests.get(url)
            json = r.json()
            for item in json['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
                driver_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
                driver_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
                driver_standings['driver'].append(item['Driver']['driverId'])
                driver_standings['driver_points'].append(int(item['points']))
                driver_standings['driver_wins'].append(int(item['wins']))
                driver_standings['driver_standings_pos'].append(int(item['position']))
    return pd.DataFrame(driver_standings)

def extract_constructor_standings(rounds):
    constructor_standings = {'season': [], 'round': [], 'constructor': [], 'constructor_points': [], 'constructor_wins': [], 'constructor_standings_pos': []}
    for year_rounds in rounds:
        year, round_list = year_rounds
        for r in round_list:
            url = f'https://ergast.com/api/f1/{year}/{r}/constructorStandings.json'
            r = requests.get(url)
            json = r.json()
            for item in json['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
                constructor_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
                constructor_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
                constructor_standings['constructor'].append(item['Constructor']['constructorId'])
                constructor_standings['constructor_points'].append(int(item['points']))
                constructor_standings['constructor_wins'].append(int(item['wins']))
                constructor_standings['constructor_standings_pos'].append(int(item['position']))
    return pd.DataFrame(constructor_standings)

def extract_qualifying_results():
    qualifying_results = pd.DataFrame()
    for year in range(2000, 2022):
        url = f'https://www.formula1.com/en/results.html/{year}/races.html'
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        year_links = [page.get('href') for page in soup.find_all('a', class_='resultsarchive-filter-item-link FilterTrigger ') if f'/en/results.html/{year}/races/' in page.get('href')]
        year_df = pd.DataFrame()
        for n, link in enumerate(year_links):
            link = link.replace('race-result.html', 'starting-grid.html')
            df = pd.read_html(f'https://www.formula1.com{link}')[0]
            df['season'] = year
            df['round'] = n + 1
            df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
            year_df = pd.concat([year_df, df])
        qualifying_results = pd.concat([qualifying_results, year_df])
    qualifying_results.rename(columns={'Pos': 'grid', 'Driver': 'driver_name', 'Car': 'car', 'Time': 'qualifying_time'}, inplace=True)
    qualifying_results.drop(columns='No', inplace=True)
    return qualifying_results

def extract_weather_data(races):
    weather = races.iloc[:, [0, 1, 2]]
    info = []
    for link in races.url:
        try:
            df = pd.read_html(link)[0]
            if 'Weather' in list(df.iloc[:, 0]):
                info.append(df.iloc[list(df.iloc[:, 0]).index('Weather'), 1])
            else:
                df = pd.read_html(link)[1]
                if 'Weather' in list(df.iloc[:, 0]):
                    info.append(df.iloc[list(df.iloc[:, 0]).index('Weather'), 1])
                else:
                    df = pd.read_html(link)[2]
                    if 'Weather' in list(df.iloc[:, 0]):
                        info.append(df.iloc[list(df.iloc[:, 0]).index('Weather'), 1])
                    else:
                        df = pd.read_html(link)[3]
                        if 'Weather' in list(df.iloc[:, 0]):
                            info.append(df.iloc[list(df.iloc[:, 0]).index('Weather'), 1])
                        else:
                            driver = webdriver.Chrome('chromedriver', options=chrome_options)
                            driver.get(link)
                            button = driver.find_element_by_link_text('Italiano')
                            button.click()
                            clima = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[1]/tbody/tr[9]/td').text
                            info.append(clima)
        except:
            info.append('not found')
    weather['weather'] = info
    weather_dict = {'weather_warm': ['soleggiato', 'clear', 'warm', 'hot', 'sunny', 'fine', 'mild', 'sereno'],
                    'weather_cold': ['cold', 'fresh', 'chilly', 'cool'],
                    'weather_dry': ['dry', 'asciutto'],
                    'weather_wet': ['showers', 'wet', 'rain', 'pioggia', 'damp', 'thunderstorms', 'rainy'],
                    'weather_cloudy': ['overcast', 'nuvoloso', 'clouds', 'cloudy', 'grey', 'coperto']}
    weather_df = pd.DataFrame(columns=weather_dict.keys())
    for col in weather_df:
        weather_df[col] = weather['weather'].map(lambda x: 1 if any(i in weather_dict[col] for i in x.lower().split()) else 0)
    weather_info = pd.concat([weather, weather_df], axis=1)
    return weather_info

def save_to_drive(df, filename):
    file = drive.CreateFile({'parents': [{'id': 'your-folder-id'}]})
    df.to_csv(filename, index=False)
    file.SetContentFile(filename)
    file.Upload()

# Extract data
races_df = extract_races_data()
save_to_drive(races_df, 'races.csv')

rounds = [(year, races_df[races_df['season'] == year]['round'].tolist()) for year in races_df['season'].unique()]
results_df = extract_results_data(rounds)
save_to_drive(results_df, 'results.csv')

driver_standings_df = extract_driver_standings(rounds)
save_to_drive(driver_standings_df, 'driver_standings.csv')

constructor_standings_df = extract_constructor_standings(rounds)
save_to_drive(constructor_standings_df, 'constructor_standings.csv')

qualifying_results_df = extract_qualifying_results()
save_to_drive(qualifying_results_df, 'qualifying_results.csv')

weather_info_df = extract_weather_data(races_df)
save_to_drive(weather_info_df, 'weather_info.csv')

# Load the data
races = pd.read_csv('races.csv')
weather = pd.read_csv('weather_info.csv')
results = pd.read_csv('results.csv')
driver_standings = pd.read_csv('driver_standings.csv')
constructor_standings = pd.read_csv('constructor_standings.csv')
qualifying = pd.read_csv('qualifying_results.csv')

# Merge dataframes
df1 = pd.merge(races, weather, how='inner', on=['season', 'round', 'circuit_id']).drop(['lat', 'long', 'country', 'weather'], axis=1)
df2 = pd.merge(df1, results, how='inner', on=['season', 'round', 'circuit_id', 'url']).drop(['url', 'points', 'status', 'time'], axis=1)
df3 = pd.merge(df2, driver_standings, how='left', on=['season', 'round', 'driver'])
df4 = pd.merge(df3, constructor_standings, how='left', on=['season', 'round', 'constructor'])

final_df = pd.merge(df4, qualifying, how='inner', on=['season', 'round', 'grid']).drop(['driver_name', 'car'], axis=1)

# Calculate driver age
from dateutil.relativedelta import relativedelta

final_df['date'] = pd.to_datetime(final_df.date)
final_df['date_of_birth'] = pd.to_datetime(final_df.date_of_birth)
final_df['driver_age'] = final_df.apply(lambda x: relativedelta(x['date'], x['date_of_birth']).years, axis=1)
final_df.drop(['date', 'date_of_birth'], axis=1, inplace=True)

# Fill/drop nulls
for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 'constructor_wins', 'constructor_standings_pos']:
    final_df[col].fillna(0, inplace=True)
    final_df[col] = final_df[col].astype(int)

final_df['qualifying_time'].fillna(0, inplace=True)
final_df.dropna(inplace=True)

# Convert to boolean to save space
for col in ['weather_warm', 'weather_cold', 'weather_dry', 'weather_wet', 'weather_cloudy']:
    final_df[col] = final_df[col].astype(bool)

# Calculate difference in qualifying times
final_df['qualifying_time'] = final_df.qualifying_time.apply(lambda x: float(x) if (":" not in str(x)) else (float(str(x).split(':')[1]) + (60 * float(str(x).split(':')[0]))))
final_df = final_df[final_df['qualifying_time'] != 0]
final_df.sort_values(['season', 'round', 'grid'], inplace=True)
final_df['qualifying_time_diff'] = final_df.groupby(['season', 'round']).qualifying_time.diff()
final_df['qualifying_time'] = final_df.groupby(['season', 'round']).qualifying_time_diff.cumsum().fillna(0)
final_df.drop('qualifying_time_diff', axis=1, inplace=True)

# Get dummies
df_dum = pd.get_dummies(final_df, columns=['circuit_id', 'nationality', 'constructor'])
for col in df_dum.columns:
    if 'nationality' in col and df_dum[col].sum() < 100:
        df_dum.drop(col, axis=1, inplace=True)
    elif 'constructor' in col and df_dum[col].sum() < 100:
        df_dum.drop(col, axis=1, inplace=True)
    elif 'circuit_id' in col and df_dum[col].sum() < 100:
        df_dum.drop(col, axis=1, inplace=True)

df_dum.to_csv('final_df.csv', index=False)
save_to_drive(df_dum, 'final_df.csv')

# Data preparation
data = pd.read_csv('final_df.csv')
df = data.copy()
df.podium = df.podium.apply(lambda x: 1 if x == 1 else 0)

# Split train data
train = df[df.season <= 2019]
X_train = train.drop(['driver', 'podium'], axis=1)
y_train = train.podium

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

# Prepare validation data
val = df[df.season == 2020]
X_val = val.drop(['driver', 'podium'], axis=1)
y_val = val.podium

X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom model architecture
class F1Model(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(F1Model, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Define pre-trained model architecture
class PretrainedModel(nn.Module):
    def __init__(self, pretrained_model, input_size, hidden_layers, dropout_rate):
        super(PretrainedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.fc = nn.Sequential(
            nn.Linear(pretrained_model.fc.in_features, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[1], 2)
        )

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        return x

# Model training and evaluation functions
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, early_stopping_patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            elif phase == 'val':
                patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model_wts)
    return model

# Hyperparameter optimization for both custom and pre-trained models
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'model_type': {'values': ['custom', 'pretrained']},
        'hidden_layers': {'values': [[64, 128], [128, 256], [256, 512]]},
        'dropout_rate': {'values': [0.2, 0.3, 0.4]},
        'learning_rate': {'values': [0.001, 0.0001, 0.00001]},
        'batch_size': {'values': [32, 64, 128]},
        'epochs': {'value': 50}
    }
}

sweep_id = wandb.sweep(sweep_config, project="f1-prediction")

def train():
    wandb.init()
    config = wandb.config

    input_size = X_train.shape[1]

    if config.model_type == 'custom':
        model = F1Model(input_size, config.hidden_layers, config.dropout_rate).to(device)
    else:
        pretrained_model = models.resnet18(pretrained=True)
        pretrained_model.fc = nn.Identity()
        model = PretrainedModel(pretrained_model, input_size, config.hidden_layers, config.dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    dataloaders = {'train': train_loader, 'val': val_loader}

    model = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=config.epochs)

    val_accuracy = evaluate_model(model, val_loader)
    wandb.log({'val_accuracy': val_accuracy})

wandb.agent(sweep_id, train, count=10)

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
    accuracy = corrects.double() / len(dataloader.dataset)
    return accuracy