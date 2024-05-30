
from google.colab import drive
drive.mount('/content/gdrive')


%mkdir gdrive/MyDrive/Colab\ Notebooks/Final_Project
%cd gdrive/MyDrive/Colab\ Notebooks/Final_Project

 [markdown]
# # Importing headers & libraries


! pip install torch_utils

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import cuda
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, sampler
import torch.nn.functional as F
from torch_utils import AverageMeter
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import inf


from sklearn.model_selection import train_test_split
import os
from glob import glob
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torch import optim, cuda, Tensor
import tqdm

# Data science tools
import numpy as np

import os

# Image manipulations
from PIL import Image
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
#plt.rcParams['font.size'] = 14

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from google.colab import files
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

!pip install selenium
!apt-get update # to update ubuntu to correctly run apt install
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

import requests
import bs4
from bs4 import BeautifulSoup
import time


file = drive.CreateFile({'parents':[{u'id': '12LkEGMmJ9602k8XZ5qtsYsmaMFeoO7Z7'}]}) 

 [markdown]
# 

 [markdown]
# # Data Processing


import pandas as pd
import numpy as np
import requests

# query API
races = {'season': [],
        'round': [],
        'circuit_id': [],
        'lat': [],
        'long': [],
        'country': [],
        'date': [],
        'url': []}

for year in list(range(2000,2022)):
    
    url = 'https://ergast.com/api/f1/{}.json'
    r = requests.get(url.format(year))
    json = r.json()

    for item in json['MRData']['RaceTable']['Races']:
        try:
            races['season'].append(int(item['season']))
        except:
            races['season'].append(None)

        try:
            races['round'].append(int(item['round']))
        except:
            races['round'].append(None)

        try:
            races['circuit_id'].append(item['Circuit']['circuitId'])
        except:
            races['circuit_id'].append(None)

        try:
            races['lat'].append(float(item['Circuit']['Location']['lat']))
        except:
            races['lat'].append(None)

        try:
            races['long'].append(float(item['Circuit']['Location']['long']))
        except:
            races['long'].append(None)

        try:
            races['country'].append(item['Circuit']['Location']['country'])
        except:
            races['country'].append(None)

        try:
            races['date'].append(item['date'])
        except:
            races['date'].append(None)

        try:
            races['url'].append(item['url'])
        except:
            races['url'].append(None)
        
races = pd.DataFrame(races)

races.to_csv('races.csv', index = False) 

file.SetContentFile("races.csv")
file.Upload()


# append the number of rounds to each season from the races_df

rounds = []
for year in np.array(races.season.unique()):
    rounds.append([year, list(races[races.season == year]['round'])])

# query API
    
results = {'season': [],
          'round':[],
           'circuit_id':[],
          'driver': [],
           'date_of_birth': [],
           'nationality': [],
          'constructor': [],
          'grid': [],
          'time': [],
          'status': [],
          'points': [],
          'podium': [],
           'url': []}

for n in list(range(len(rounds))):
    for i in rounds[n][1]:
    
        url = 'http://ergast.com/api/f1/{}/{}/results.json'
        r = requests.get(url.format(rounds[n][0], i))
        json = r.json()

        for item in json['MRData']['RaceTable']['Races'][0]['Results']:
            try:
                results['season'].append(int(json['MRData']['RaceTable']['Races'][0]['season']))
            except:
                results['season'].append(None)

            try:
                results['round'].append(int(json['MRData']['RaceTable']['Races'][0]['round']))
            except:
                results['round'].append(None)

            try:
                results['circuit_id'].append(json['MRData']['RaceTable']['Races'][0]['Circuit']['circuitId'])
            except:
                results['circuit_id'].append(None)

            try:
                results['driver'].append(item['Driver']['driverId'])
            except:
                results['driver'].append(None)
            
            try:
                results['date_of_birth'].append(item['Driver']['dateOfBirth'])
            except:
                results['date_of_birth'].append(None)
                
            try:
                results['nationality'].append(item['Driver']['nationality'])
            except:
                results['nationality'].append(None)

            try:
                results['constructor'].append(item['Constructor']['constructorId'])
            except:
                results['constructor'].append(None)

            try:
                results['grid'].append(int(item['grid']))
            except:
                results['grid'].append(None)

            try:
                results['time'].append(int(item['Time']['millis']))
            except:
                results['time'].append(None)

            try:
                results['status'].append(item['status'])
            except:
                results['status'].append(None)

            try:
                results['points'].append(int(item['points']))
            except:
                results['points'].append(None)

            try:
                results['podium'].append(int(item['position']))
            except:
                results['podium'].append(None)
            try:
                results['url'].append(json['MRData']['RaceTable']['Races'][0]['url'])
            except:
                results['url'].append(None)

           
results = pd.DataFrame(results)
results.to_csv('results.csv', index = False)
file.SetContentFile("results.csv")
file.Upload()


driver_standings = {'season': [],
                    'round':[],
                    'driver': [],
                    'driver_points': [],
                    'driver_wins': [],
                   'driver_standings_pos': []}

# query API

for n in list(range(len(rounds))):     
    for i in rounds[n][1]:    # iterate through rounds of each year
    
        url = 'https://ergast.com/api/f1/{}/{}/driverStandings.json'
        r = requests.get(url.format(rounds[n][0], i))
        json = r.json()

        for item in json['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
            try:
                driver_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
            except:
                driver_standings['season'].append(None)

            try:
                driver_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
            except:
                driver_standings['round'].append(None)
                                         
            try:
                driver_standings['driver'].append(item['Driver']['driverId'])
            except:
                driver_standings['driver'].append(None)
            
            try:
                driver_standings['driver_points'].append(int(item['points']))
            except:
                driver_standings['driver_points'].append(None)
            
            try:
                driver_standings['driver_wins'].append(int(item['wins']))
            except:
                driver_standings['driver_wins'].append(None)
                
            try:
                driver_standings['driver_standings_pos'].append(int(item['position']))
            except:
                driver_standings['driver_standings_pos'].append(None)
            
driver_standings = pd.DataFrame(driver_standings)

# define lookup function to shift points and number of wins from previous rounds

def lookup (df, team, points):
    df['lookup1'] = df.season.astype(str) + df[team] + df['round'].astype(str)
    df['lookup2'] = df.season.astype(str) + df[team] + (df['round']-1).astype(str)
    new_df = df.merge(df[['lookup1', points]], how = 'left', left_on='lookup2',right_on='lookup1')
    new_df.drop(['lookup1_x', 'lookup2', 'lookup1_y'], axis = 1, inplace = True)
    new_df.rename(columns = {points+'_x': points+'_after_race', points+'_y': points}, inplace = True)
    new_df[points].fillna(0, inplace = True)
    return new_df
  
driver_standings = lookup(driver_standings, 'driver', 'driver_points')
driver_standings = lookup(driver_standings, 'driver', 'driver_wins')
driver_standings = lookup(driver_standings, 'driver', 'driver_standings_pos')

driver_standings.drop(['driver_points_after_race', 'driver_wins_after_race', 'driver_standings_pos_after_race'], 
                      axis = 1, inplace = True)

driver_standings.to_csv('driver_standings.csv', index = False)
file.SetContentFile("driver_standings.csv")
file.Upload()


# start from year 2000

constructor_rounds = rounds[8:]

constructor_standings = {'season': [],
                    'round':[],
                    'constructor': [],
                    'constructor_points': [],
                    'constructor_wins': [],
                   'constructor_standings_pos': []}
# query API

for n in list(range(len(constructor_rounds))):
    for i in constructor_rounds[n][1]:
    
        url = 'https://ergast.com/api/f1/{}/{}/constructorStandings.json'
        r = requests.get(url.format(constructor_rounds[n][0], i))
        json = r.json()

        for item in json['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
            try:
                constructor_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
            except:
                constructor_standings['season'].append(None)

            try:
                constructor_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
            except:
                constructor_standings['round'].append(None)
                                         
            try:
                constructor_standings['constructor'].append(item['Constructor']['constructorId'])
            except:
                constructor_standings['constructor'].append(None)
            
            try:
                constructor_standings['constructor_points'].append(int(item['points']))
            except:
                constructor_standings['constructor_points'].append(None)
            
            try:
                constructor_standings['constructor_wins'].append(int(item['wins']))
            except:
                constructor_standings['constructor_wins'].append(None)
                
            try:
                constructor_standings['constructor_standings_pos'].append(int(item['position']))
            except:
                constructor_standings['constructor_standings_pos'].append(None)
            
constructor_standings = pd.DataFrame(constructor_standings)

constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_points')
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_wins')
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_standings_pos')

constructor_standings.drop(['constructor_points_after_race', 'constructor_wins_after_race','constructor_standings_pos_after_race' ],
                           axis = 1, inplace = True)
constructor_standings.to_csv('constructor_standings.csv', index = False)

file.SetContentFile("constructor_standings.csv")
file.Upload()


qualifying_results = pd.DataFrame()

# Qualifying times are only available from 2000

for year in list(range(2000,2022)):
    url = 'https://www.formula1.com/en/results.html/{}/races.html'
    r = requests.get(url.format(year))
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # find links to all circuits for a certain year
    
    year_links = []
    #<a href="/en/results.html/2022/races.html" class="resultsarchive-filter-item-link FilterTrigger " data-name="year" data-value="2022">
    for page in soup.find_all('a', attrs = {'class':"resultsarchive-filter-item-link FilterTrigger "}):
      
        link = page.get('href')
        
        if f'/en/results.html/{year}/races/' in link: 
            year_links.append(link)
    
    # for each circuit, switch to the starting grid page and read table

    year_df = pd.DataFrame()
    new_url = 'https://www.formula1.com{}'
    for n, link in list(enumerate(year_links)):
        link = link.replace('race-result.html', 'starting-grid.html')
        df = pd.read_html(new_url.format(link))
        df = df[0]
        df['season'] = year
        df['round'] = n+1
        for col in df:
            if 'Unnamed' in col:
                df.drop(col, axis = 1, inplace = True)

        year_df = pd.concat([year_df, df])

    # concatenate all tables from all years  
        
    qualifying_results = pd.concat([qualifying_results, year_df])

# rename columns
    
qualifying_results.rename(columns = {'Pos': 'grid', 'Driver': 'driver_name', 'Car': 'car',
                                     'Time': 'qualifying_time'}, inplace = True)
# drop driver number column

qualifying_results.drop('No', axis = 1, inplace = True)

qualifying_results.to_csv('qualifying_results.csv', index = False)

file.SetContentFile("qualifying_results.csv")
file.Upload()


weather = races.iloc[:,[0,1,2]]

info = []

# read wikipedia tables

for link in races.url:
    try:
        df = pd.read_html(link)[0]
        if 'Weather' in list(df.iloc[:,0]):
            n = list(df.iloc[:,0]).index('Weather')
            info.append(df.iloc[n,1])
        else:
            df = pd.read_html(link)[1]
            if 'Weather' in list(df.iloc[:,0]):
                n = list(df.iloc[:,0]).index('Weather')
                info.append(df.iloc[n,1])
            else:
                df = pd.read_html(link)[2]
                if 'Weather' in list(df.iloc[:,0]):
                    n = list(df.iloc[:,0]).index('Weather')
                    info.append(df.iloc[n,1])
                else:
                    df = pd.read_html(link)[3]
                    if 'Weather' in list(df.iloc[:,0]):
                        n = list(df.iloc[:,0]).index('Weather')
                        info.append(df.iloc[n,1])
                    else:
                        driver = webdriver.Chrome()
                        driver.get(link)

                        # click language button
                        button = driver.find_element_by_link_text('Italiano')
                        button.click()
                        
                        # find weather in italian with selenium
                        
                        clima = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[1]/tbody/tr[9]/td').text
                        info.append(clima) 
                                
    except:
        info.append('not found')

# append column with weather information to dataframe  
  
weather['weather'] = info

# set up a dictionary to convert weather information into keywords

weather_dict = {'weather_warm': ['soleggiato', 'clear', 'warm', 'hot', 'sunny', 'fine', 'mild', 'sereno'],
               'weather_cold': ['cold', 'fresh', 'chilly', 'cool'],
               'weather_dry': ['dry', 'asciutto'],
               'weather_wet': ['showers', 'wet', 'rain', 'pioggia', 'damp', 'thunderstorms', 'rainy'],
               'weather_cloudy': ['overcast', 'nuvoloso', 'clouds', 'cloudy', 'grey', 'coperto']}

# map new df according to weather dictionary

weather_df = pd.DataFrame(columns = weather_dict.keys())
for col in weather_df:
    weather_df[col] = weather['weather'].map(lambda x: 1 if any(i in weather_dict[col] for i in x.lower().split()) else 0)
   
weather_info = pd.concat([weather, weather_df], axis = 1)

weather_info.to_csv('weather_info.csv', index = False)

file.SetContentFile("weather_info.csv")
file.Upload()


races = pd.read_csv('races.csv')
weather = pd.read_csv('weather_info.csv')
results = pd.read_csv('results.csv')
driver_standings = pd.read_csv('driver_standings.csv')
constructor_standings = pd.read_csv('constructor_standings.csv')
qualifying = pd.read_csv('qualifying_results.csv')


qualifying.rename(columns = {'grid_position': 'grid'}, inplace = True)


df1 = pd.merge(races, weather, how='inner', on=['season', 'round', 'circuit_id']).drop(['lat', 'long','country','weather'], axis = 1)
df2 = pd.merge(df1, results, how='inner', on=['season', 'round', 'circuit_id', 'url']).drop(['url','points', 'status', 'time'], axis = 1)

df3 = pd.merge(df2, driver_standings, how='left', on=['season', 'round', 'driver']) 
df4 = pd.merge(df3, constructor_standings, how='left', on=['season', 'round', 'constructor']) #from 1958

final_df = pd.merge(df4, qualifying, how='inner', on=['season', 'round', 'grid']).drop(['driver_name', 'car'], axis = 1) #from 1983


# calculate age of drivers
from dateutil.relativedelta import *
final_df['date'] = pd.to_datetime(final_df.date)
final_df['date_of_birth'] = pd.to_datetime(final_df.date_of_birth)
final_df['driver_age'] = final_df.apply(lambda x: 
                                        relativedelta(x['date'], x['date_of_birth']).years, axis=1)
final_df.drop(['date', 'date_of_birth'], axis = 1, inplace = True)


# fill/drop nulls

for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 
            'constructor_wins' , 'constructor_standings_pos']:
    final_df[col].fillna(0, inplace = True)
    final_df[col] = final_df[col].map(lambda x: int(x))


final_df['qualifying_time'].fillna(0, inplace = True)
final_df.dropna(inplace = True )


# convert to boolean to save space

for col in ['weather_warm', 'weather_cold','weather_dry', 'weather_wet', 'weather_cloudy']:
    final_df[col] = final_df[col].map(lambda x: bool(x))


# calculate difference in qualifying times
final_df['qualifying_time'] = final_df.qualifying_time.map( lambda x: float(x) if((":" in str(x))== False)
                             else(float(str(x).split(':')[1]) + 
                                  (60 * float(str(x).split(':')[0])) if x != 0 else 0))


final_df = final_df[final_df['qualifying_time'] != 0]



final_df.sort_values(['season', 'round', 'grid'], inplace = True)
final_df['qualifying_time_diff'] = final_df.groupby(['season', 'round']).qualifying_time.diff()
final_df['qualifying_time'] = final_df.groupby(['season', 'round']).qualifying_time_diff.cumsum().fillna(0)
final_df.drop('qualifying_time_diff', axis = 1, inplace = True)

final_df.head()



# get dummies

df_dum = pd.get_dummies(final_df, columns = ['circuit_id', 'nationality', 'constructor'] )

for col in df_dum.columns:
    if 'nationality' in col and df_dum[col].sum() < 100:
        df_dum.drop(col, axis = 1, inplace = True)
        
    elif 'constructor' in col and df_dum[col].sum() < 100:
        df_dum.drop(col, axis = 1, inplace = True)
        
    elif 'circuit_id' in col and df_dum[col].sum() < 100:
        df_dum.drop(col, axis = 1, inplace = True)
    
    else:
        pass


df_dum.to_csv('final_df.csv', index = False)


file.SetContentFile("final_df.csv")
file.Upload()

 [markdown]
# #  Data Preperation 


data = pd.read_csv('final_df.csv')


from sklearn.preprocessing import StandardScaler
df = data.copy()
df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)

#split train

train = df[df.season <= 2019]
X_train = train.drop(['driver', 'podium'], axis = 1)
y_train = train.podium

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

print(X_train.head)
print(X_train.shape)
print(y_train.shape)


X_train = X_train.to_numpy()
y_train = y_train.to_numpy()


print(X_train.shape[1])
print(y_train.shape)


from sklearn.metrics import precision_score
def score_classification_2020(model):
    score = 0
    for circuit in df[df.season == 2020]['round'].unique():

        test = df[(df.season == 2020) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium

        #scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == 2020]['round'].unique().max()
    return model_score

 [markdown]
# # Multi-layered Neural Network + weights & biasses API
# 


import matplotlib.pyplot as plt
import torch
from torch import cuda
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, sampler
import torch.nn.functional as F
from torch_utils import AverageMeter
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import inf


from sklearn.model_selection import train_test_split
import os
from glob import glob
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torch import optim, cuda, Tensor
import tqdm

# Data science tools
import numpy as np

import os

# Image manipulations
from PIL import Image
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
#plt.rcParams['font.size'] = 14

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

 [markdown]
# ## WandB API


!pip install wandb -qqq
import wandb


!pip install wandb --upgrade


# Log in to your W&B account
wandb.login()



sweep_config = {
    'method': 'random'
    }


metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric




parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size1': {
        'values': [65,70, 75, 80]
        },
        'fc_layer_size2': {
        'values': [15, 20, 25, 30]
        },
        'fc_layer_size3': {
        'values': [40, 50, 60]
        },

        'fc_layer_size4': {
        'values': [5, 8, 10]
        },
        'dropout': {
          'values': [0.1,0.15,0.2,0.25,0.3]
        },
        'criterion': {
        'values': ['crossentropy']
        }
    }

sweep_config['parameters'] = parameters_dict


parameters_dict.update({
    'epochs': {
        'value': 2}
    })


parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.3
      },
    'batch_size': {
        #'value': 1
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q':  2,
        'min': 2,
        'max': 32,
      }
    })


import pprint

pprint.pprint(sweep_config)


sweep_id = wandb.sweep(sweep_config, project="Formula One Predictor")

 [markdown]
# ###Network Design


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

 [markdown]
# ## Data Loaders


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader, shape = build_dataset(config.batch_size)
        network = build_network(shape,config.fc_layer_size1,config.fc_layer_size2,config.fc_layer_size3,config.fc_layer_size4, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        #lossfcn = config.lossfcn
        for epoch in range(config.epochs):
            avg_loss,network_trained = train_epoch(network, loader, optimizer, config.criterion)
            wandb.log({"loss": avg_loss, "epoch": epoch})
        score_2020 = score_classification_2020_nn(network_trained, config.criterion)
        score_2021 = score_classification_2021_nn(network_trained, config.criterion)
        wandb.log({"model accuracy (2020) (%)": score_2020*100})
        wandb.log({"model accuracy (2021) (%)": score_2021*100})    
        wandb.log({"score 2020": score_2020,  "score 2021": score_2021})       


from sklearn.metrics import precision_score
def score_classification_2020_nn(model, criterion1):
    score = 0
    valid_loss = 0.0
    
    
    with torch.no_grad():
        model.eval()
        for circuit in df[df.season == 2020]['round'].unique():

            test = df[(df.season == 2020) & (df['round'] == circuit)]
            
            
            X_test = test.drop(['driver', 'podium'], axis = 1)
            y_test = test.podium

            
            

            #scaling
            X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
            X_test = X_test.to_numpy()
            y_test1 = y_test.to_numpy()
            X_test = torch.from_numpy(X_test).float()
            y_test1 = torch.from_numpy(y_test1).to(torch.long)
        
            data = X_test.to(device)
            target = y_test1.to(device)
            # make predictions
            output = model(data)
            
            _, pred = torch.max(output, dim=0)
            output = output.cpu().data.numpy()
            pred = pred.cpu().data.numpy()
            
            prediction_df = pd.DataFrame(output, columns = ['proba_0',  'proba_1'])
            
         
            prediction_df['driver'] = test['driver'].reset_index(drop = True)
            prediction_df['actual'] = y_test.reset_index(drop = True)
            prediction_df.sort_values('proba_1', ascending = False, inplace = True)
            prediction_df.reset_index(inplace = True, drop = True)
            prediction_df['predicted'] = prediction_df.index
            prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
            
      
            score += precision_score(prediction_df.actual, prediction_df.predicted)

        model_score = score / df[df.season == 2020]['round'].unique().max()
        valid_loss = valid_loss/df[df.season == 2020]['round'].unique().max()
    return model_score

def score_classification_2021_nn(model, criterion):
    score = 0
    valid_loss = 0.0
    with torch.no_grad():
        
        for circuit in df[df.season == 2021]['round'].unique():

            test = df[(df.season == 2021) & (df['round'] == circuit)]
            
            
            X_test = test.drop(['driver', 'podium'], axis = 1)
            y_test = test.podium

            #scaling
            X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
            X_test = X_test.to_numpy()
            y_test1 = y_test.to_numpy()
            X_test = torch.from_numpy(X_test).float()
            y_test1 = torch.from_numpy(y_test1)

            data = X_test.to(device)
            target = y_test1.to(device)
            # make predictions
            output = model(data)
            
            _, pred = torch.max(output, dim=0)
            output = output.cpu().data.numpy()
            pred = pred.cpu().data.numpy()
            
            prediction_df = pd.DataFrame(output, columns = ['proba_0',  'proba_1'])
            
            
            prediction_df['driver'] = test['driver'].reset_index(drop = True)
            prediction_df['actual'] = y_test.reset_index(drop = True)
            prediction_df.sort_values('proba_1', ascending = False, inplace = True)
            prediction_df.reset_index(inplace = True, drop = True)
            prediction_df['predicted'] = prediction_df.index
            prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
            print(prediction_df)
      
            score += precision_score(prediction_df.actual, prediction_df.predicted)

        model_score = score / df[df.season == 2021]['round'].unique().max()
        valid_loss = valid_loss/df[df.season == 2021]['round'].unique().max()
    return model_score




def build_dataset(batch_size):
    data = {
    'train':
    TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).to(torch.long))}

    dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,num_workers=2)}
    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    features.shape, labels.shape
    print(features.shape)
    return dataloaders['train'], features.shape

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def build_lossfcn(criterion, network, data, target):
    if criterion == "nll":
        criterion = F.nll_loss(network(data), target)
        loss = criterion
    elif criterion == "crossentropy":
        criterion = nn.CrossEntropyLoss()
        loss = criterion(network(data), target)
    elif criterion == "mse":
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(network(data), target)
    return criterion

def build_network(shape,fc_layer_size1,fc_layer_size2,fc_layer_size3,fc_layer_size4, dropout):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Flatten(),
        nn.Linear(shape[1], fc_layer_size1), 
        
        nn.Linear(fc_layer_size1, fc_layer_size2), 
        
        nn.Linear(fc_layer_size2, fc_layer_size3), 
        
        nn.Linear(fc_layer_size3, fc_layer_size4), 
        
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size4, 2),
        nn.Sigmoid()
    )
    print(shape[0]* shape[1])

    return network.to(device)

def train_epoch(network, loader, optimizer, criterion1):
    cumu_loss = 0
    
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass

        
        if criterion1 == "nll":
            criterion = F.nll_loss(network(data), target)
            loss = criterion
        elif criterion1 == "crossentropy":
            criterion = nn.CrossEntropyLoss()
            loss = criterion(network(data), target)
        
        cumu_loss += loss.item()
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        
        #wandb.log({"validation loss": valid_loss})
        
        wandb.log({"batch loss": loss.item()})
        loss =(cumu_loss / len(loader))
    
    return loss,network


wandb.agent(sweep_id, train, count=5)

 [markdown]
# # MLPClassifier + weights & biasses API


df = data.copy()
df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)

#split train

train = df[df.season <2019]
X_train = train.drop(['driver', 'podium'], axis = 1)
y_train = train.podium

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)


sweep_config = {
    'method': 'random'
    }


metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric


parameters_dict = {
    'activation': {
        'values': ['identity', 'logistic', 'tanh', 'relu']
        },
    'solver': {
        'values': ['lbfgs', 'sgd', 'adam']
        },
    'hidden_layer1': {
        'values': [65,70, 75, 80]
        },
        'hidden_layer2': {
        'values': [15, 20, 25, 30]
        },
        'hidden_layer3': {
        'values': [40, 50, 60]
        },

        'hidden_layer4': {
        'values': [5, 8, 10]
        },
    }

sweep_config['parameters'] = parameters_dict


parameters_dict.update({
    'epochs': {
        'value': 1}
    })


parameters_dict.update({
    'alpha': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    
    })


sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")


import pprint

pprint.pprint(sweep_config)


from sklearn.metrics import precision_score
def score_classification_2021(model):
    score = 0
    for circuit in df[df.season == 2021]['round'].unique():

        test = df[(df.season == 2021) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium

        #scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        prediction_df['driver'] = test['driver'].reset_index(drop = True)
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == 2021]['round'].unique().max()
    return model_score


from sklearn.metrics import precision_score
def score_classification_2020(model):
    score = 0
    for circuit in df[df.season == 2020]['round'].unique():

        test = df[(df.season == 2020) & (df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium

        #scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        prediction_df['driver'] = test['driver'].reset_index(drop = True)
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == 2020]['round'].unique().max()
    return model_score


from sklearn.neural_network import MLPClassifier
def train_epoch_2(hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4, activation, solver, alpha):
    
    model2 = MLPClassifier(hidden_layer_sizes = (hidden_layer1,hidden_layer1,hidden_layer1,hidden_layer1),
                          activation = activation, solver = solver, alpha = alpha, random_state = 1)
    model2.fit(X_train, y_train)
    loss = model2.loss_
    return loss, model2


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        #lossfcn = config.lossfcn
        for epoch in range(config.epochs):
            loss, model= train_epoch_2(config.hidden_layer1, config.hidden_layer2, config.hidden_layer3, config.hidden_layer4, config.activation, config.solver, config.alpha)

        score_2020 = score_classification_2020(model)
        score_2021 = score_classification_2021(model)
        wandb.log({"loss":loss, "epoch": epoch, "score_2020": score_2020*100, "score_2021": score_2021*100})      


wandb.agent(sweep_id, train, count=5)


