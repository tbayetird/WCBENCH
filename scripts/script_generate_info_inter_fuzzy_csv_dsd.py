import os
import json
import pandas as pd
import numpy as np

path_to_dataset = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\DollarStreetDataset\\dataset_dollarstreet\\images_benchmark.csv"
dsd_info = pd.read_csv(path_to_dataset)
path_to_prediction = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\DollarStreetDataset\\dataset_dollarstreet\\Benchmark_data\\prediction_CSRA.csv"
prediction_df = pd.read_csv(path_to_prediction)

DSD_WESTERN_COUNTRIES = ['Serbia','Ukraine','United States','France','Netherlands',
'Austria','Sweden','United Kingdom','Romania','Switzerland',
'Spain','Czech Republic','Canada','Italy','Denmark']

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

label_match = ['car', 'bed', 'toilet', 'tv', 'oven', 'refrigerator', 'toothbrush']

fuzzy_matching = {
                'person':['teeth','hand palm','hand back'],
                'bicycle':['bike'],
                'car':['car'],
                'motorcycle':['moped/motorcycle'],
                'airplane':[],
                'bus':[],
                'train':[],
                'truck':[],
                'boat':[],
                'traffic_light':[],
                'fire_hydrant':[],
                'stop_sign':[],
                'parking_meter':[],
                'bench':[],
                'bird':['chickens'],
                'cat':[],
                'dog':[],
                'horse':[],
                'sheep':[],
                'cow':[],
                'elephant':[],
                'bear':[],
                'zebra':[],
                'giraffe':[],
                'backpack':[],
                'umbrella':[],
                'handbag':[],
                'tie':[],
                'suitcase':[],
                'frisbee':[],
                'skis':[],
                'snowboard':[],
                'sports_ball':[],
                'kite':[],
                'baseball_bat':[],
                'baseball_glove':[],
                'skateboard':[],
                'surfboard':[],
                'tennis_racket':[],
                'bottle':['alcoholic drinks'],
                'wine_glass':[],
                'cup':['cups/mugs/glasses'],
                'fork':[],
                'knife':['knifes'],
                'spoon':[],
                'bowl':[],
                'banana':[],
                'apple':[],
                'sandwich':[],
                'orange':[],
                'broccoli':[],
                'carrot':[],
                'hot_dog':[],
                'pizza':[],
                'donut':[],
                'cake':[],
                'chair':[],
                'couch':['couch','sofa'],
                'potted_plant':[],
                'bed':['bedroom','bed kids','guest bed','bed','children room'],
                'dining_table':['table with food'],
                'toilet':['bathroom/toilet','toilet'],
                'tv':['tv'],
                'laptop':['computer'],
                'mouse':[],
                'remote':[],
                'keyboard':[],
                'cell_phone':['phone'],
                'microwave':[],
                'oven':['oven','stove/hob'],
                'toaster':[],
                'sink':['kitchen sink'],
                'refrigerator':['refrigerator','freezer'],
                'book':['books'],
                'clock':['arm watch','wall clock'],
                'vase':[],
                'scissors':[],
                'teddy_bear':[],
                'hair_drier':[],
                'toothbrush':['toothbrush','toothpaste on toothbrush']
}

fuzzy_labels = [key for key,val in fuzzy_matching.items() if val != []]
print(fuzzy_labels)
## ['person', 'bicycle', 'car', 'motorcycle', 'bird', 'bottle', 'cup', 'knife', 'couch', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'cell_phone', 'oven', 'sink', 'refrigerator', 'book', 'clock', 'toothbrush']
print(len(fuzzy_labels))



'''
Réalise l'intersection entre un label et une liste de classes autorisées

* annotations : str comprenant les annotations sous format ann1,ann2,..

'''
def intersection(annotations):
    annotations = annotations.split(', ')
    return [x for x in annotations if x in label_match]

'''
Réalise l'intersection entre une prediction et une liste de classes autorisées

* annotations : str comprenant les annotations sous format ann1,ann2,..

'''
def intersection_prediction(annotations,label_match):
    if type(annotations) is not list :
        annotations = annotations[1:-1].replace('\'','')
        annotations = annotations.split(', ')
    # print(annotations)
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
    annotations = annotations.split(',')
    fuzzy_ann = []
    for ann in annotations :
        for key,value in fuzzy_matching.items() :
            if ann in value :
                fuzzy_ann.append(key)
    return fuzzy_ann

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

# print(dsd_info['topics'].unique())

### format
## intersection
dsd_info['ID']=dsd_info['id']
dsd_info['imgPath']=dsd_info['imageRelPath'].apply(lambda x: os.path.join(os.path.dirname(path_to_dataset),x))
dsd_info['annotation']=dsd_info['topics'].apply(intersection)
dsd_info['occident'] = dsd_info['country.name'].apply(lambda x : x in DSD_WESTERN_COUNTRIES)
dsd_info['ann_logit'] = dsd_info['annotation'].apply(ann2logit)
dsd_info['prediction']=prediction_df['prediction_CSRA'].apply(logit2ann)
dsd_info['prediction']=dsd_info['prediction'].apply(intersection_prediction,label_match=label_match)
dsd_info['pred_logit']=dsd_info['prediction'].apply(ann2logit_prediction,label_match=label_match)

output_df = dsd_info[['ID','imgPath','annotation','ann_logit','occident','prediction','pred_logit']]
output_df.to_csv(os.path.join(os.path.dirname(path_to_dataset),"Benchmark_data\\info_CSRA_intersection.csv"))

## fuzzy matching
dsd_info['annotation']=dsd_info['topics'].apply(fuzzy_match)
dsd_info['ann_logit']=dsd_info['annotation'].apply(fuzzy_ann2logit)
dsd_info['prediction']=prediction_df['prediction_CSRA'].apply(logit2ann).apply(intersection_prediction,label_match=fuzzy_labels)
dsd_info['pred_logit']=dsd_info['prediction'].apply(ann2logit_prediction,label_match=fuzzy_labels)


output_df = dsd_info[['ID','imgPath','annotation','ann_logit','occident','prediction','pred_logit']]
output_df.to_csv(os.path.join(os.path.dirname(path_to_dataset),"Benchmark_data\\info_CSRA_fuzzy.csv"))
