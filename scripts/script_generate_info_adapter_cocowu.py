import os,sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LabelAdapter.label_adapter import LabelAdapter

## CONFIG
model='Google'
model_names = ["Google","Amazon","Microsoft"]

### USEFUL DATA and PARAMS
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

occidental_zones = ['Australia_and_New_Zealand','Eastern_Europe','Northern_America','Northern_Europe','Southern_Europe','Western_Europe']

batch_size=32
epoch = 1000

## FUNCTIONS
def get_adapter(coco_classes,model_classes,model_pred_path,batch_size,epoch,adapter_model_path):
    LA = LabelAdapter(coco_classes,model_classes,model_pred_path,batch_size,epoch)
    if not os.path.exists(adapter_model_path):
        print("No model found, training for {} epochs ".format(epoch))
        LA.train()
        LA.save_model(adapter_model_path)
    else:
        print("Found a model, loading model")
        LA.load_model(adapter_model_path)
    return LA

def adapt_prediction(pred, LA=None):
    return LA.get_prediction(pred)

def pred_to_logit(pred):
    logits = [int(x) for x in [elem in pred for elem in coco_classes ]]


### RUN
## run one model
def run_one_conf(model):
    print("Working on model {}".format(model))
    ### PATH and LOADS
    model_pred_path = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\{}\\prediction_{}.csv".format(model,model.upper())
    prediction_df = pd.read_csv(model_pred_path)
    model_classes = pd.read_csv("C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\Benchmark\\scripts\\{}_preds_labels.csv".format(model.upper()),encoding = "ISO-8859-1")
    model_classes = model_classes['classes'].values.tolist()
    model_classes = [x.lower() for x in model_classes]
    adapter_model_path = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\{}\\adapter_{}.pt".format(model,model.upper())
    output_df = pd.DataFrame([],columns = ['ID','imgPath','url','annotation','ann_logit','occident','prediction','pred_logit'])

    ## Model training / accessing
    LA = get_adapter(coco_classes,model_classes,model_pred_path,batch_size,epoch,adapter_model_path)

    ## Generating output
    output_df['ID']=prediction_df['ID']
    output_df['imgPath']=prediction_df['imgPath']
    output_df['annotation']=prediction_df['annotation']
    output_df['ann_logit'] = prediction_df['annotation']
    output_df['url']=prediction_df['url']
    output_df['occident']=prediction_df['occident']
    output_df['prediction']=prediction_df['prediction'].apply(adapt_prediction,LA=LA)
    output_df['pred_logit'] = prediction_df['prediction'].apply(pred_to_logit)

    output_df.to_csv(os.path.join(os.path.dirname(model_pred_path),"info_{}_adapter.csv".format(model.upper())))

def run_all(model_names):
    for model_name in model_names:
        run_one_conf(model_name)

run_all(model_names)
