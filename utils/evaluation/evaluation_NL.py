import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
# from r import krippalpha

def accuracy_at_5_label(label,pred):
    pred = pred[:5]
    if label in pred :
        return 1
    else:
        return 0

def accuracy_at_5(ann,pred):
    # print(ann)
    if ann==['']:
        return 1.0
    if len(pred)==0:
        return 0.0
    acc_list = []
    for label in ann:
        acc_list.append(accuracy_at_5_label(label,pred))
    return np.mean(acc_list)

def accuracy_per_class(logit_ann,logit_pred,class_i):
    if logit_ann[class_i]==0:
        TODO
        pass

def from_ann_to_list(ann):
    if ann == '[]':
        return []
    ann_out = ann[1:-1].replace('\'','').split(', ')
    return ann_out


class Eval_NL :

    def __init__(self,conf):
        self._load_conf(conf)


    """
    Loading the configuration. TODO : describe the configuration file
    """
    def _load_conf(self,conf):
        self.MATCH_CLS = conf.MATCH_CLS
        pass

    """
    Computes HL and ACC for two annotations

    """
    def evaluate(self,ann,pred):
        # print(ann)
        ann = ann[1:-1].replace('\'','')
        ann = ann.split(', ')
        # print('after treatment : ',ann,len(ann))
        pred = pred[1:-1].replace('\'','')
        pred = pred.split(', ')
        logit_true = self._from_ann_to_logit(ann,self.MATCH_CLS)
        logit_pred = self._from_ann_to_logit(pred,self.MATCH_CLS)
        HL = hamming_loss(logit_true,logit_pred)
        ACC = accuracy_score(logit_true,logit_pred)
        ACC5 = accuracy_at_5(ann,pred)
        F1 = f1_score(logit_true,logit_pred,average='weighted')
        return HL,ACC,ACC5,F1

    """
    Computes the logits for a given label matching the target_list annotations

    * ann : the annotations
    * target_list : list of classes used for the logits
    """
    def _from_ann_to_logit(self,ann,target_list):
        logits = np.zeros(len(target_list))
        for label in ann:
            if label in target_list :
                logits[[i for i,elem in enumerate(target_list) if elem==label][0]]=1
        return logits


    """
    Computes the evaluation of a dataframe with "annotation" and "prediction" columns
    Return a DataFrame with the two columns HL and ACC
    """
    def compute_evaluation(self,data_df):
        assert 'annotation' in data_df, 'Format error when computing evaluation, no annotation column found '
        assert 'prediction' in data_df, 'Format error when computing evaluation, no prediction column found '
        output_df = pd.DataFrame([],columns = ['HL','ACC','ACC5','F1'])
        for row in data_df.iterrows() :
            row = row[1]
            HL, ACC, ACC5,F1 = self.evaluate(row['annotation'],row['prediction'])
            output_row = pd.DataFrame([[HL,ACC,ACC5,F1]],columns=['HL','ACC','ACC5','F1'])
            output_df = pd.concat([output_df,output_row],ignore_index=True)
        return output_df

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
        mean_ACC5_occ = np.mean(df_occ['ACC5'])
        mean_ACC5_not_occ = np.mean(df_not_occ['ACC5'])
        mean_F1_occ = np.mean(df_occ['F1'])
        mean_F1_not_occ = np.mean(df_not_occ['F1'])
        return [mean_HL_occ,mean_HL_not_occ], [mean_acc_occ,mean_acc_not_occ],[mean_ACC5_occ,mean_ACC5_not_occ],[mean_F1_occ,mean_F1_not_occ]



    '''
    Computes the score for the benchmark, ie. the HL and ACC scores for occidental
    and non occidental zones / countries. The function sorts out all the data
    with empty labels.

    * data_df : DataFrame with columns "occident","ACC","HL", and "anotation" at least.
    occident column should have true or false.
    '''
    def compute_non_empty_benchmark_scores(self,data_df):
        data_df = data_df.loc[data_df['annotation']!='[]',:]
        return self.compute_benchmark_scores(data_df)


####################
##### PER CLASS
####################
    """
    Computes ACC_per_class for a single annotation with given class to evaluate

    """
    def evaluate_per_class(self,ann,pred,class_index):
        # print(ann)
        ann = ann[1:-1].replace('\'','')
        ann = ann.split(', ')
        # print('after treatment : ',ann,len(ann))
        pred = pred[1:-1].replace('\'','')
        pred = pred.split(', ')
        logit_true = [self._from_ann_to_logit(ann,self.MATCH_CLS)[class_index]]
        logit_pred = [self._from_ann_to_logit(pred,self.MATCH_CLS)[class_index]]

        ACC = accuracy_score(logit_true,logit_pred)
        return ACC


    """
    Computes the per class accuracy of a dataframe with "annotation" and "prediction" columns
    Return a DataFrame with the three columns 'class','count' and 'ACC'
    """
    def compute_per_class_evaluation(self,data_df):
        assert 'annotation' in data_df, 'Format error when computing evaluation, no annotation column found '
        assert 'prediction' in data_df, 'Format error when computing evaluation, no prediction column found '
        output_df = pd.DataFrame([],columns = ['class','count','ACC'])
        for class_index,class_name in enumerate(self.MATCH_CLS):

            #TODO : récupérer les booléens pour les deux colomnes, puis les combiner dans le .loc .
            bool_ann = data_df['annotation'].apply(lambda x : class_name in from_ann_to_list(x))
            bool_pred = data_df['prediction'].apply(lambda x : class_name in from_ann_to_list(x))
            class_data_df=data_df.loc[bool_ann+bool_pred]
            if len(class_data_df)==0:
                class_score=0
                class_count=0
            else :
                cumul_ACC = []
                for row in class_data_df.iterrows() :
                    row = row[1]
                    # print(row)
                    ACC = self.evaluate_per_class(row['annotation'],row['prediction'],class_index)
                    cumul_ACC.append(ACC)
                class_score=np.mean(cumul_ACC)
                class_count = len(class_data_df)
            output_row = pd.DataFrame([[class_name,class_count,class_score]],columns=['class','count','ACC'])
            output_df = pd.concat([output_df,output_row],ignore_index=True)
        return output_df

    '''
    Computes the mean per class accuracy given a dataframe with the columns
    count and ACC.
    '''
    def compute_per_class_mean_acc(self,data_df):
        if len(data_df)==0:
            return 0
        data_df['weighted_acc']=data_df['count']*data_df['ACC']
        sum_acc = np.sum(data_df['weighted_acc'])
        data_count = np.sum(data_df['count'])
        return sum_acc/data_count

    '''
    Computes the per class accuracy of a dataframe with "annotation",
    "prediction" and "occident" colums, sorted by boolean in occident.
    Return a DataFrame with the columns 'class','occ_count','occ_ACC',
    'not_occ_count' and not_occ_ACC

    * data_df : DataFrame with columns "occident", "ACC", "HL" at least. occ should
    be True or False in each row
    '''
    def compute_per_class_benchmark_scores(self,data_df):
        df_occ = data_df.loc[data_df['occident'].apply(lambda x : x)]
        df_not_occ = data_df.loc[data_df['occident'].apply(lambda x : not bool(x))]

        classes_all = self.compute_per_class_evaluation(data_df)

        classes_occ = self.compute_per_class_evaluation(df_occ)
        mean_acc_occ=self.compute_per_class_mean_acc(classes_occ)

        classes_not_occ = self.compute_per_class_evaluation(df_not_occ)
        mean_acc_not_occ=self.compute_per_class_mean_acc(classes_not_occ)
        return classes_occ,classes_not_occ,classes_all,[mean_acc_occ,mean_acc_not_occ]

    '''
    Computes the per class score for the benchmark, ie. ACC scores for occidental
    and non occidental zones / countries. The function sorts out all the data
    with empty labels.

    * data_df : DataFrame with columns "occident","ACC","HL", and "anotation" at least.
    occident column should have true or false.
    '''
    def compute_per_class_non_empty_benchmark_scores(self,data_df):
        data_df = data_df.loc[data_df['annotation']!='[]',:]
        return self.compute_per_class_benchmark_scores(data_df)
