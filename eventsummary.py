import pandas as pd
import json


def get_eventdict(eventdata,gamedata):
    '''
    With given gamedata and eventdata, return 2 dictionaries
    make_qeventdict and miss_qeventdict, which contains event id and attacker of the event by quart(1,2,3,4)
    
    params
    eventdata : csv formant, columns with eventnum, eventmsgtype, and team id 
    gamedata : json format, Main game information data to know which team is home or visit
    '''
    
    df = pd.read_csv(eventdata)
    df = df.sort_values(by='EVENTNUM')

    with open(gamedata,'r') as f:
        data = json.load(f)
    hometeam = data['events'][0]['home']['abbreviation']
    visitorteam = data['events'][0]['visitor']['abbreviation']

    quarter_eventdict={}
    qstart=0
    qend=0
    qcount = 1
    for pos, x in enumerate(list(df['EVENTMSGTYPE'])):
        if x == 12:
            qstart = pos+1
        elif x== 13:
            qend = pos      
        if qstart !=0 and qend !=0:
            quarter_eventdict['Q%s'%(qcount)] = list(df['EVENTNUM'])[qstart:qend]
            qstart=0
            qend=0
            qcount +=1

    makeevent = df[df['EVENTMSGTYPE']==1]
    missevent = df[df['EVENTMSGTYPE']==2]

    make_qeventdict={}
    for k in quarter_eventdict.keys():
        make_qeventdict[k]=[]
        for pos, x in enumerate(list(makeevent['EVENTNUM'])):
            if x in quarter_eventdict[k]:
                t = list(makeevent['TEAM'])[pos]
                
                if t ==hometeam:
                    make_qeventdict[k].append([x,'HOM'])
                else:
                    make_qeventdict[k].append([x,'VIS'])

    miss_qeventdict={}
    for k in quarter_eventdict.keys():
        miss_qeventdict[k]=[]
        for pos,x in enumerate(list(missevent['EVENTNUM'])):
            if x in quarter_eventdict[k]:
                t = list(missevent['TEAM'])[pos]
                if t==hometeam:
                    miss_qeventdict[k].append([x,'HOME'])
                else:
                    miss_qeventdict[k].append([x,'VIS'])

    return make_qeventdict, miss_qeventdict

if __name__ =='__main__':
    eventdata = './game_eventdata.csv'
    gamedata = './data/0021500507.json'
    print(get_eventdict(eventdata,gamedata))
