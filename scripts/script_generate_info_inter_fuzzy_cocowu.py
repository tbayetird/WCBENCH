import os,sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fuzzy_matching import amazon_to_coco,google_to_coco,microsoft_to_coco

#PARAM
model='empty'

## CONFIG
if model=='Amazon':
    model_fuzzy_matching = amazon_to_coco
if model=='Google':
    model_fuzzy_matching = google_to_coco
if model=='Microsoft':
    model_fuzzy_matching = microsoft_to_coco


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


model_pred_path = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\{}\\prediction_{}.csv".format(model,model.upper())
prediction_df = pd.read_csv(model_pred_path)
if model=='empty':
    model_classes=coco_classes
    model_fuzzy_matching={}
else :

    model_classes = pd.read_csv("C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\Benchmark\\scripts\\{}_preds_labels.csv".format(model.upper()),encoding = "ISO-8859-1")
    model_classes = model_classes['classes'].values.tolist()
    model_classes = [x.lower() for x in model_classes]
output_df = pd.DataFrame([],columns = ['ID','imgPath','url','annotation','ann_logit','occident','prediction','pred_logit'])



occidental_zones = ['Australia_and_New_Zealand','Eastern_Europe','Northern_America','Northern_Europe','Southern_Europe','Western_Europe']

## FUNCTIONS

'''
Réalise l'intersection entre un label et une liste de classes autorisées

* annotations : str comprenant les annotations sous format ann1,ann2,..

'''
def intersection(annotations,label_match):
    annotations = annotations[1:-1].replace('\'','')
    annotations = annotations.split(', ')
    # print(annotations)
    return [x.lower() for x in annotations if x.lower() in label_match]

'''
From annotations to logits

*annotations : label as a list of classes associated with an image
*label_match : list of labels to use as a base for the logits
'''
def ann2logit(annotations,label_match,verbose=0):
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
Réalise le fuzzy matching entre une annotation et une liste comprenant les
clés des matchs entre label set d'annotations et prédictions.

* annotations : str comprenant les annotations sous format cls1,cls2,..
* fuzzy_labels : labels retenus pour le fuzzy matching
'''
def fuzzy_match_annotations(annotations,fuzzy_labels):
    if annotations=="[]":
        return []
    annotations = annotations[1:-1].replace('\'','').replace(' ','')
    annotations = annotations.split(',')
    fuzzy_ann = []
    for ann in annotations :
        if ann in fuzzy_labels:
            fuzzy_ann.append(ann)
    return fuzzy_ann

'''
Réalise le fuzzy matching entre une prediction et un dictionnaire comprenant les
matchs entre label set d'annotations et prédictions. Le dictionnaire a pour clé
les labels d'annotation et pour value les prédictions correspondantes.

* predictions : str comprenant les annotations sous format cls1,cls2,..
* fuzzy_matching : dictionnaire pour le matching entre annotations et predictions
'''

def fuzzy_match_predictions(predictions,fuzzy_matching):
    if predictions=="[]":
        return []
    predictions = predictions[1:-1].replace('\'','').replace(' ','')
    predictions = predictions.split(',')
    fuzzy_pred = []
    for pred in predictions :
        # print('pred')
        for key,value in fuzzy_matching.items() :
            if pred.lower() in [val.lower() for val in value] :
                fuzzy_pred.append(key)
    return list(set(fuzzy_pred))


'''
From fuzzy annotations to fuzzy logits

* annotations : list of labels associated with an image
'''
def fuzzy_ann2logit(annotations,fuzzy_labels):
    if type(annotations) is not list :
        annotations = annotations[1:-1].replace('\'','').replace(' ','')
        annotations = annotations.split(',')
    logit_ann = np.zeros(len(fuzzy_labels))
    for label in annotations:
        label=label.lower()
        # print(label)
        if label in fuzzy_labels :
            logit_ann[[i for i,elem in enumerate(fuzzy_labels) if elem==label][0]]=1
    return logit_ann

## UTILS

model_intersection = [x.lower() for x in model_classes if x.lower() in coco_classes]
fuzzy_labels = [key for key,val in model_fuzzy_matching.items() if val != []]

# print('model intersection : ', model_intersection)
# # print([x for x in coco_classes if x not in model_intersection])
#
print("About Fuzzy matching : ")
print(" Classes from dataset with at least one match :")
print(fuzzy_labels)
print('Length : ',len(fuzzy_labels))
### RUN
## intersection
output_df['ID']=prediction_df['ID']
output_df['imgPath']=prediction_df['imgPath']
output_df['annotation']=prediction_df['annotation'].apply(intersection,label_match=(model_intersection))
output_df['ann_logit'] = prediction_df['annotation'].apply(ann2logit,label_match=(model_intersection))
output_df['url']=prediction_df['url']
output_df['occident']=prediction_df['occident']
output_df['prediction']=prediction_df['prediction'].apply(intersection,label_match=(model_intersection))
output_df['pred_logit'] = prediction_df['prediction'].apply(ann2logit,label_match=(model_intersection))

output_df.to_csv(os.path.join(os.path.dirname(model_pred_path),"info_{}_intersection.csv".format(model.upper())))

## fuzzy matching
output_df['annotation']=prediction_df['annotation'].apply(fuzzy_match_annotations,fuzzy_labels=fuzzy_labels)
output_df['ann_logit']=prediction_df['annotation'].apply(fuzzy_ann2logit,fuzzy_labels = fuzzy_labels)
output_df['prediction']=prediction_df['prediction'].apply(fuzzy_match_predictions,fuzzy_matching=(model_fuzzy_matching))
output_df['pred_logit'] = output_df['prediction'].apply(fuzzy_ann2logit,fuzzy_labels=fuzzy_labels)

output_df.to_csv(os.path.join(os.path.dirname(model_pred_path),"info_{}_fuzzy.csv".format(model.upper())))
