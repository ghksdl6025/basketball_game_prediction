# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pickle
import gzip
import eventsummary
from tqdm import tqdm
import logging
from playsound import playsound

def eventsearcher(eventid,attacker):
    '''
    paras
    eventid : id of the event
    attacker : Between home and visitor which team attacks

    return
    Moment in event dictionary. Key is moment, value = ball_loc, attacker_loc, defender_loc
    '''

    momentdict={}
    count = 0

    for loc in data['events'][eventid]['moments']:   
        locations = loc[5]
        ball_loc= (locations[0][2],locations[0][3])
        home_player = [(x[2],x[3]) for x in locations[1:6]]
        visit_player = [(x[2],x[3]) for x in locations[6:11]]

        if attacker == hometeam:
            momentdict[count] = (ball_loc,home_player,visit_player)
        else:
            momentdict[count] = (ball_loc,visit_player,home_player)
        count +=1
    
    return momentdict

    
def draw_all_position_jpg(momentdict,eventid):
    '''
    Plot jpg 3 file which records ball, attacker team, and defender team location by each.
    However, jpg file has 3 channel 'RGB', these jpg files will be converted to 277*515*1 array only to get location

    For the radius of the player circle, assume average of nba player arm span is 210cm. And take the half of it as circle diameter.
    Since court width size in axis is 100, So relative radius of circle is approximately 3.75.   
    '''

    count =0
    for moment in momentdict.keys():
        dirpath = './data/game_data/'+game+'/event_'+str(eventid)+'/'
        try:
            os.makedirs(dirpath)
        except:
            pass

        fig =plt.figure(1)
        ax1 = plt.subplot()
        
        img  = mpimg.imread('./data/nba_court.jpg')
        axis =[0,100,0,50]
        ax1.imshow(img,extent=axis,zorder =0)
        ax1 = plt.gca()
        plt.cla()

        ball_circle = plt.Circle(momentdict[moment][0],2,color = 'green')
        ax1.add_patch(ball_circle)
        plt.axis('off')
        dx =2
        plt.xlim([0-dx,100+dx])
        plt.ylim([0-dx,50+dx])
        plt.savefig(dirpath+'ball_loc'+str(moment)+'.jpg', bbox_inches='tight')
        plt.cla()

        for attack in momentdict[moment][1]:
            att_circle = plt.Circle(attack,3.75,color = 'blue')
            ax1.add_patch(att_circle)
        
        plt.axis('off')
        dx =2
        plt.xlim([0-dx,100+dx])
        plt.ylim([0-dx,50+dx])
        plt.savefig(dirpath+'attacker_loc'+str(moment)+'.jpg', bbox_inches='tight')
        plt.cla()
        
        for defender in momentdict[moment][2]:
            def_circle = plt.Circle(defender,3.75,color = 'red')
            ax1.add_patch(def_circle)
    
        plt.axis('off')
        dx =2
        plt.xlim([0-dx,100+dx])
        plt.ylim([0-dx,50+dx])
        plt.savefig(dirpath+'defender_loc'+str(moment)+'.jpg', bbox_inches='tight')
        
        plt.cla()
        plt.clf()


def jpg_to_array(eventid):
    
    courtjpg = Image.open('./data/nba_court.jpg')
    courtarr = np.array(courtjpg)[:,:,0]
    dirpath = './data/game_data/'+game+'/event_'+str(eventid)+'/'
    momentlen = int(len(os.listdir(dirpath))/3)
    for moment in range(momentlen):
        if os.path.isfile(dirpath+'ball_loc'+str(moment)+'.jpg') and os.path.isfile(dirpath+'attacker_loc'+str(moment)+'.jpg') and os.path.isfile(dirpath+'defender_loc'+str(moment)+'.jpg'):
            balljpg = Image.open(dirpath+'ball_loc'+str(moment)+'.jpg')
            attjpg = Image.open(dirpath+'attacker_loc'+str(moment)+'.jpg')
            defjpg = Image.open(dirpath+'defender_loc'+str(moment)+'.jpg')

            ball2arr = np.array(balljpg)[:,:,0]
            att2arr = np.array(attjpg)[:,:,1]
            def2arr = np.array(defjpg)[:,:,2]

            allarr = np.array([courtarr,ball2arr,att2arr,def2arr])
            
            try:
                os.makedirs(dirpath)
            except:
                pass

            with gzip.open(dirpath+'moment'+str(moment)+'.pkl','wb')  as f:
                pickle.dump(allarr,f)

def delete_img(eventid):
    dirpath = './data/game_data/'+game+'/event_'+str(eventid)+'/'
    imgs = [x for x in os.listdir(dirpath) if x.endswith('.jpg')]
    for img in imgs:
        os.remove(dirpath+img)

if __name__=='__main__':

    
    filelist = os.listdir('./data/game_data/')
    filelist = [x.split('.')[0] for x in filelist if x.endswith('.json')]
    
    for game in tqdm(filelist[15:]):
        game = str(game)
        gamedata = './data/game_data/'+game+'.json'
        eventdata = './data/game_data/game_event/game_eventdata_'+game+'.csv'

        with open(gamedata,'r') as f:
            data = json.load(f)
        hometeam = data['events'][0]['home']['abbreviation']
        visitorteam = data['events'][0]['visitor']['abbreviation']

        eventlocdict={}
        for t in range(len(data['events'])):
            eventlocdict[int(data['events'][t]['eventId'])]=t
        
        make_eventdict, miss_eventdict = eventsummary.get_eventdict(eventdata,gamedata)

        for q in ['Q2','Q3','Q4']:
            for events in tqdm(list(miss_eventdict[q])):  
                momentdict = eventsearcher(eventlocdict[events[0]],events[1])
                if bool(momentdict):
                    draw_all_position_jpg(momentdict,events[0])
                    jpg_to_array(events[0])
                    delete_img(events[0])
    playsound('c:/Users/suhwanlee/Downloads/Yattong edited version.mp3')