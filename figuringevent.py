import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import os

filelist = os.listdir('./data/game_data/test/')
filelist = [x.split('.')[0] for x in filelist]
# filelist = filelist[12:-1]
for gamecode in filelist:

    game_url ='https://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID='+str(gamecode)+'&RangeType=2&Season=2015-16&SeasonType=Regular+Season&StartPeriod=1&StartRange=0'
    headers = {
        'Referer': 'https://stats.nba.com/game/'+str(gamecode)+'/playbyplay/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true'
    }
    response = requests.get(game_url,headers=headers,timeout=5)
    json_obj = response.json()

    headers = json_obj['resultSets'][0]['headers']
    gameData = json_obj['resultSets'][0]['rowSet']
    df = pd.DataFrame(gameData, columns=headers) #turn the data into a pandas dataframe
    df = df[[df.columns[1], df.columns[2],df.columns[7],df.columns[9],df.columns[18]]] #there's a ton of data here, so I trim  it doown
    df['TEAM'] = df['PLAYER1_TEAM_ABBREVIATION']
    df = df.drop('PLAYER1_TEAM_ABBREVIATION', 1)

    df.to_csv('./data/game_data/test/game_eventdata_'+str(gamecode)+'.csv',index=False)
