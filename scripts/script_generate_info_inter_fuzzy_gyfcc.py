import os
import json
import pandas as pd
import numpy as np

path_to_predictions = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\GeoYFCC\\Benchmark_data\\CSRA\\CSRA_predictions.csv"
preds = pd.read_csv(path_to_predictions)

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

occidental_zones = ['Bulgaria','Croatia','Switzerland','Israel','Netherlands',
'Ukraine','Italy','Germany','Sweden','Ireland','Australia','New Zealand','Finland',
'United Kingdom','Canada','United States','Iceland','Portugal','Romania','Austria',
'Greece','Poland','Czech Republic','Spain','Belgium','France','Norway','Hungary',
'Denmark']

# generated thanks to the script_explore_geo_yfcc
# same for the GYFCC_preds_labels.csv file
model_intersection=['boat', 'book', 'remote', 'clock', 'car', 'orange', 'elephant', 'sheep', 'bottle', 'bed', 'truck', 'apple', 'skateboard', 'bench', 'snowboard', 'chair', 'cup', 'bowl', 'toaster', 'umbrella', 'banana', 'oven', 'teddy_bear', 'keyboard', 'sandwich', 'spoon', 'couch', 'knife', 'vase', 'tie', 'broccoli']

fuzzy_matching={
                'person':['human','kid','man','mechanical_man'],
                'bicycle':['bike'],
                'car':['ambulance','car','motorcar','passenger_vehicle','pickup_truck','suv','sport_car','racing_car','stock_car','squad_car'],
                'motorcycle':['sidecar','scooter'],
                'airplane':['aircraft','plane','jet-propelled_plane','attack_aircraft','hydroplane'],
                'bus':['minibus'],
                'train':['trolley_car','railroad_train'],
                'truck':['truck','fire_truck'],
                'boat':['boat','ship','ferryboat','pirate_ship','powerboat','watercraft','sailing_ship','fireboat','u-boat','combat_ship','speedboat','attack_aircraft_carrier','lifeboat'],
                'traffic_light':[],
                'fire_hydrant':['hydrant'],
                'stop_sign':[],
                'parking_meter':[],
                'bench':['bench'],
                'bird':[],
                'cat':[],
                'dog':[],
                'horse':['pony'],
                'sheep':['sheep'],
                'cow':[],
                'elephant':['elephant'],
                'bear':[],
                'zebra':[],
                'giraffe':['giraffa_camelopardalis'],
                'backpack':[],
                'umbrella':['umbrella'],
                'handbag':[],
                'tie':['black_tie','tie'],
                'suitcase':[],
                'frisbee':[],
                'skis':['ski'],
                'snowboard':['snowboard'],
                'sports_ball':['ball'],
                'kite':[],
                'baseball_bat':[],
                'baseball_glove':[],
                'skateboard':['skate','skateboard'],
                'surfboard':[],
                'tennis_racket':[],
                'bottle':['bottle'],
                'wine_glass':['wineglass'],
                'cup':['cup','mug'],
                'fork':[],
                'knife':['knife'],
                'spoon':['spoon'],
                'bowl':['bowl'],
                'banana':['banana'],
                'apple':['apple'],
                'sandwich':['sandwich'],
                'orange':['orange'],
                'broccoli':['broccoli'],
                'carrot':[],
                'hot_dog':[],
                'pizza':['pizza_pie'],
                'donut':[],
                'cake':[],
                'chair':['chair'],
                'couch':['couch'],
                'potted_plant':['flowerpot'],
                'bed':['bed','bedchamber'],
                'dining_table':[],
                'toilet':[],
                'tv':['television_system'],
                'laptop':['laptop_computer','notebook_computer'],
                'mouse':['computer_mouse'],
                'remote':['remote'],
                'keyboard':['keyboard'],
                'cell_phone':['phone'],
                'microwave':[],
                'oven':['stove','oven','cooking_stove'],
                'toaster':['toaster'],
                'sink':[],
                'refrigerator':['fridge'],
                'book':['book'],
                'clock':['clock'],
                'vase':['vase'],
                'scissors':[],
                'teddy_bear':['teddy_bear'],
                'hair_drier':[],
                'toothbrush':[]
}


fuzzy_labels = [key for key,val in fuzzy_matching.items() if val != []]
print(fuzzy_labels)
## ['person', 'bicycle', 'car', 'motorcycle', 'bird', 'bottle', 'cup', 'knife', 'couch', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'cell_phone', 'oven', 'sink', 'refrigerator', 'book', 'clock', 'toothbrush']
print(len(fuzzy_labels))



## FUNCTIONS

'''
Réalise l'intersection entre un label et une liste de classes autorisées

* annotations : str comprenant les annotations sous format ann1,ann2,..

'''
def intersection(annotations,label_match):
    annotations = annotations.replace('[','').replace(']','').replace('\'','').split(', ')
    # [print(x) for x in annotations]
    return [x for x in annotations if x in label_match]

'''
Réalise l'intersection entre une prediction et une liste de classes autorisées

* annotations : str comprenant les annotations sous format ann1,ann2,..

'''
def intersection_prediction(annotations,label_match):
    if type(annotations) is not list :
        annotations = annotations[1:-1].replace('\'','')
        annotations = annotations.split(', ')
    # [print(x) for x in annotations]
    return [x.lower() for x in annotations if x.lower() in label_match]

'''
From annotations to logits

* annotations : list of labels associated with an image
'''
def ann2logit(annotations):
    logit_ann = np.zeros(len(label_match))
    for label in annotations:
        if label in label_match :
            logit_ann[[i for i,elem in enumerate(label_match) if elem==label][0]]=1
    return logit_ann

'''
From annotations to logits

*annotations : label as a list of classes associated with an image
*label_match : list of labels to use as a base for the logits
'''
def ann2logit_prediction(annotations,label_match,verbose=0):
    if type(annotations) is not list :
        annotations = annotations[1:-1].replace('\'','')
        annotations = annotations.split(', ')
    logit_ann = np.zeros(len(label_match))
    for label in annotations:
        label=label.lower()
        if verbose==1:
            print(label)
        if label in label_match :
            logit_ann[[i for i,elem in enumerate(label_match) if elem==label][0]]=1
    return logit_ann


'''
From logits to annotations

*logits : logits a priori from the CSRA predictions. Using coco classes.
'''
def logit2ann(logits):
    logits = logits[1:-1].replace('.','')
    logits = logits.split(' ')
    logits=[int(x) for x in logits]
    # print("Len logits : ",len(logits))
    annotations=[]
    # print(logits)
    for i,elem in enumerate(logits):
        if elem==1:
            annotations.append(coco_classes[i])
    # print(annotations)
    return annotations

'''
Réalise le fuzzy matching entre un label et un dictionnaire comprenant les matchs
entre DSD et MS COCO

* annotations : str comprenant les annotations sous format cls1,cls2,..
'''
def fuzzy_match(annotations):
    if type(annotations) is not list :
        annotations = annotations[1:-1].replace('\'','')
        annotations = annotations.split(', ')
    # print(annotations)
    # annotations = annotations.split(',')
    fuzzy_ann = []
    for ann in annotations :
        for key,value in fuzzy_matching.items() :
            if ann in value :
                fuzzy_ann.append(key)
    return list(set(fuzzy_ann))

'''
From fuzzy annotations to fuzzy logits

* annotations : list of labels associated with an image
'''
def fuzzy_ann2logit(annotations):
    logit_ann = np.zeros(len(fuzzy_labels))
    for label in annotations:
        if label in fuzzy_labels :
            logit_ann[[i for i,elem in enumerate(fuzzy_labels) if elem==label][0]]=1
    return logit_ann


## RUN
intersection_df = preds.copy()
intersection_df['annotation']=intersection_df['annotation'].apply(intersection,label_match=(model_intersection))
intersection_df['prediction']=intersection_df['prediction'].apply(intersection,label_match=(model_intersection))
intersection_df.to_csv(os.path.join(os.path.dirname(path_to_predictions),"info_CSRA_intersection.csv"))

fuzzy_df = preds.copy()
# fuzzy_df['annotation']=fuzzy_df['annotation'].apply(fuzzy_match_annotations,fuzzy_labels=fuzzy_labels)
# fuzzy_df['prediction']=fuzzy_df['prediction'].apply(fuzzy_match_predictions,fuzzy_matching=(fuzzy_matching))


## fuzzy matching
fuzzy_df['annotation']=preds['annotation'].apply(fuzzy_match)
fuzzy_df['prediction']=preds['prediction'].apply(intersection_prediction,label_match=fuzzy_labels)
fuzzy_df.to_csv(os.path.join(os.path.dirname(path_to_predictions),"info_CSRA_fuzzy.csv"))
