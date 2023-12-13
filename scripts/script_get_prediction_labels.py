import os
import pandas as pd

prediction_path = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\prediction_AMAZON.csv"
output_path = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\Benchmark\\scripts\\AMAZON_preds_labels.txt"

'''
Return the cleaned list of annotations from the DSD dataset

*dsd_data : the dsd dataframe
'''
def get_topics(data_df):
    topics=[]
    for topic_list in data_df['prediction'].unique() :
        cleaned_topics = clean_topics(topic_list)
        for topic in cleaned_topics:
            if topic not in topics:
                topics.append(topic)
    return topics


'''
Cleaning process for the classes of the DSD dataset. From raw list to list of topics.

* raw_list : raw list from the dsd dataframe
'''
def clean_topics(raw_list):
    if raw_list=="[]":
        return []
    clean_topics =  raw_list.replace('\'','').replace('[','').replace(']','').split(',')
    clean_topics = list(map(lambda x:x[:-1] if x[-1]==" " else x,clean_topics))
    clean_topics = list(map(lambda x:x[1:] if x[0]==" " else x,clean_topics))
    clean_topics = [topic.lower() for topic in clean_topics]
    return clean_topics


prediction_df = pd.read_csv(prediction_path)
predictions = get_topics(prediction_df)
print(len(predictions))

with open(output_path,'w') as file :
    file.write('classes\n')
    for elem in predictions :
        try :
            file.write(elem+'\n')
        except UnicodeEncodeError:
            print(f' Unicode error because of {elem}')
