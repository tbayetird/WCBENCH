import os
import json
import pandas as pd
import numpy as np

path_to_dataset = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\"
path_to_imgs = os.path.join(path_to_dataset,"Images")
path_to_annotations = os.path.join(path_to_dataset,"Annotations")
path_to_Licenses = os.path.join(path_to_dataset,"Licenses")

output_df = pd.DataFrame([],columns = ['ID','imgPath','url','annotation','ann_logit','occident'])

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


'''
From "script_from_csv_to_hamming_loss"
Permet de lire un dictionnaire d'annotations extraite par la méthode
'load_annotations_as_dic', et de restreindre ces annotations à une liste précise.
Ces annotations comprennent un ensemble d'images et sont lues images par images.
Cette fonction renvoie un dictionnaire liant à une image son annotation sous
forme de logit (suivant la liste fournie)

* annotations : le dictionnaire comprenant les annotations.
* label_match : la liste de labels à laquelle on restreint les annotations

'''
def _load_annotations_constrained_list(annotations,label_match=coco_classes):
    logit_dic = {}
    for key in annotations.keys():
        logit_ann = np.zeros(len(label_match))
        list_ann=[]
        image_ann = annotations[key]
        attributes = image_ann['file_attributes']
        image_name = image_ann['filename']
        #navigate through the attributes dic
        for att_key in attributes.keys():
            #att_key are the 11 groups of the categories
            category_dic = attributes[att_key]
            if category_dic!= {}:
                for category in category_dic.keys():
                    category=category.lower()
                    # those are the annotation of the image
                    # check if the given labels are in the constraining label list
                    if category in label_match :
                        list_ann.append(category)
                        # print(image_name + ' : ' + category )
                        logit_ann[[i for i,elem in enumerate(label_match) if elem==category][0]]=1
        logit_dic[image_name]=(list_ann,logit_ann)
    return logit_dic

'''
Grab the url for the image from the correct license file

'''
def get_url(img_ID,dirname):
    for root,dir,files in os.walk(path_to_Licenses):
        for file in files :
            if dirname in file :
                license_file = file
                break
    license_df = pd.read_csv(os.path.join(path_to_Licenses,license_file))
    if license_df.empty :
        print(img_ID, dirname)
        return 'Unknown'
    id = img_ID.split('-')[-1].split('_')[0]
    # print(id)
    url = license_df.loc[license_df['id'].apply(lambda x : x == int(id))]['url']
    # print(url)
    return list(url)[0]



### Script run ###

for root,dirs,files in os.walk(path_to_imgs):
    for dir_name in dirs :
        if "400" not in dir_name :
            continue
        # print(dir_name)
        annotation_name = "annotations_" + dir_name + ".json"
        dir_annotations = json.load(open(os.path.join(path_to_annotations,annotation_name)))
        dir_ann_logits = _load_annotations_constrained_list(dir_annotations)
        dir_img_list = os.listdir(os.path.join(root,dir_name))
        occ = dir_name.replace('_400','') in occidental_zones
        # print(dir_ann_logits)
        # print(dir_img_list)
        for img_name in dir_img_list :
            img_path = os.path.join(root,dir_name,img_name)
            list_ann,logit = dir_ann_logits[img_name]
            url = get_url(img_name,dir_name.replace('_400',''))
            img_row = pd.DataFrame([[img_name,img_path,url,list_ann,logit,occ]],columns=['ID','imgPath','url','annotation','ann_logit','occident'])
            output_df = pd.concat([output_df,img_row],ignore_index=True)

    break
output_df.to_csv(os.path.join(path_to_dataset,"Benchmark_data\\info_AMAZON.csv"))
