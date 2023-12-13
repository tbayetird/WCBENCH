import os,sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fuzzy_matching import amazon_to_coco,google_to_coco,microsoft_to_coco

#PARAM
model='CSRA'

coco_classes = ['person', 'car', 'bird']

model_pred_path = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\{}\\prediction_{}.csv".format(model,model.upper())
prediction_df = pd.read_csv(model_pred_path)
if model=='empty' or model=='CSRA':
    model_classes=coco_classes
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


## UTILS

model_intersection = [x.lower() for x in model_classes if x.lower() in coco_classes]

# print('model intersection : ', model_intersection)
# # print([x for x in coco_classes if x not in model_intersection])
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

output_df.to_csv(os.path.join(os.path.dirname(model_pred_path),"info_{}_intersection_three_classes.csv".format(model.upper())))
