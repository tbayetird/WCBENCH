import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score

class Stats :

    def __init__(self,conf):
        pass

    '''
    Return the cleaned list of topics from the DSD dataset in the report_CSRA
    dataFrame

    *data_df : a DatraFrame with the topics, in a column namned 'annotations'
    '''
    def get_topics(self,dsd_data):
        topics=[]
        for topic_list in dsd_data['annotation'].unique() :
            cleaned_topics = self.clean_topics(topic_list)
            for topic in cleaned_topics:
                if topic not in topics:
                    topics.append(topic)
        return topics


    '''
    Cleaning process for the classes of the DSD dataset. From raw list to list of topics.

    * raw_list : raw list from the annotations stored in the report dataFrame
    '''
    def clean_topics(self,raw_list):
        if raw_list=='[]':
            return []
        clean_list = raw_list[1:-1].split(',')
        return clean_list

    '''
    To be used in a .apply method. Returns True if the arg topic is in the list.

    * topics_raw_list : row information from the dsd dataset
    * topic : class name we want to find in the rows of the dataframe

    '''
    def is_topic_in(self,topics_raw_list,topic) :
         topics_list = self.clean_topics(topics_raw_list)
         return (topic in topics_list)

    '''
    To be used in a .apply method. Returns True if the arg id is in the list.

    * row_ID : row information from the ID column of the DataFrame
    * ID_list : list of accepted IDs

    '''
    def is_id_in(self,row_ID,ID_list) :
        return (row_ID in list(ID_list))


    """
    Computes HL and ACC for two logits

    """
    def evaluate(self,logit_true,logit_pred):
        HL = hamming_loss(logit_true,logit_pred)
        ACC = accuracy_score(logit_true,logit_pred)
        return HL,ACC

    '''
    Computes the score for the benchmark, ie. the HL and ACC scores for occidental
    and non occidental zones / countries.

    * data_df : DataFrame with columns "occident", "ACC", "HL" at least. occ should
    be True or False in each row
    '''
    def compute_benchmark_scores(self,data_df):
        df_occ = data_df.loc[data_df['occident'].apply(lambda x : x)]
        df_not_occ = data_df.loc[data_df['occident'].apply(lambda x : not x)]
        mean_acc_occ = np.mean(df_occ['ACC'])
        mean_acc_not_occ = np.mean(df_not_occ['ACC'])
        mean_HL_occ = np.mean(df_occ['HL'])
        mean_HL_not_occ = np.mean(df_not_occ['HL'])
        return [mean_HL_occ,mean_HL_not_occ], [mean_acc_occ,mean_acc_not_occ]

    '''
    Computes benchmark scores for each class, and store the results in a
    DataFrame.
    '''
    def compute_DSD_class_stats(self,data_df):
        class_dicts=[]
        for class_name in self.get_topics(data_df):
            class_dict={'CLS_NAME':class_name}
            class_df =  data_df.loc[data_df['annotation'].apply(self.is_topic_in,args=(class_name,))]
            HL,ACC = self.compute_benchmark_scores(class_df)
            class_dict['HL_occ']= HL[0]
            class_dict['HL_not_occ']= HL[1]
            class_dict['ACC_occ']= ACC[0]
            class_dict['ACC_not_occ']= ACC[1]
            class_dicts.append(class_dict)
        out_df = pd.DataFrame(class_dicts)
        return out_df

    '''
    Computes benchmark scores for each country, and store the results in a
    DataFrame.
    '''
    def compute_DSD_country_stats(self,data_df,dsd_df):
        country_dicts=[]
        for country_name in dsd_df['country.name'].unique():
            country_dict={'COUNTRY':country_name}
            country_dsd_df = dsd_df.loc[dsd_df['country.name']==country_name]
            country_ids = country_dsd_df['id']
            country_df =  data_df.loc[data_df['ID'].apply(self.is_id_in,args=(country_ids,))]
            HL,ACC = self.compute_benchmark_scores(country_df)
            country_dict['HL_occ']= HL[0]
            country_dict['HL_not_occ']= HL[1]
            country_dict['ACC_occ']= ACC[0]
            country_dict['ACC_not_occ']= ACC[1]
            country_dicts.append(country_dict)
        out_df = pd.DataFrame(country_dicts)
        return out_df

    '''
    Computes the score for the benchmark, ie. the HL and ACC scores for occidental
    and non occidental zones / countries. The function sorts out all the data
    with empty labels.

    * data_df : DataFrame with columns "occident","ACC","HL", and "anotation" at least.
    occident column should have true or false.
    '''
    def compute_non_empty_country_stats(self,data_df,dsd_df):
        data_df = data_df.loc[data_df['annotation']!='[]',:]
        return self.compute_DSD_country_stats(data_df,dsd_df)
