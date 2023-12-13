import sys, os
import pandas as pd
import warnings

class OutputsWriter :

    def __init__(self,conf):
        self._load_conf(conf)

    def _load_conf(self,conf):
        self.report_csv_path=conf.REPORT_CSV_PATH
        self.benchmark_csv_path=conf.BENCHMARK_CSV_PATH

    '''
    Writes the final report CSV with all results and metrics and a benchmark report
    '''
    def write_report(self,output_df,results_HL,results_ACC, results_ACC5, results_F1):
        output_df.to_csv(self.report_csv_path,index=False)
        benchmark_df = pd.DataFrame([],columns = ['Zone','HL','ACC','ACC5','F1'])
        # results_occ.insert(0,'Western')
        # results_not_occ.insert(0,'Non Western')
        # benchmark_df=pd.concat([benchmark_df,pd.DataFrame([results_occ],columns = ['Zone','HL','ACC'])],ignore_index=True)
        # benchmark_df=pd.concat([benchmark_df,pd.DataFrame([results_not_occ],columns = ['Zone','HL','ACC'])],ignore_index=True)
        zone_data = ['Western','Non Western','Western non empty', 'Non Western non empty']
        benchmark_df['Zone']=zone_data
        benchmark_df['HL']=results_HL
        benchmark_df['ACC']=results_ACC
        benchmark_df['ACC5']=results_ACC5
        benchmark_df['F1']=results_F1
        benchmark_df.to_csv(self.benchmark_csv_path,index=False)


    '''
    Writes the final per class report CSV with all results and metrics and a benchmark report
    '''
    def write_per_class_report(self,results_ACC):
        benchmark_df = pd.DataFrame([],columns = ['Zone','per_class_ACC'])
        zone_data = ['Western','Non Western','Western non empty', 'Non Western non empty']
        benchmark_df['Zone']=zone_data
        benchmark_df['per_class_ACC']=results_ACC
        benchmark_df.to_csv(self.benchmark_csv_path,index=False)



    '''
    Gather all the data we have and produce a csv with everything in it
    '''
    def prepare_output_df(self,df_info, df_prediction, df_eval):
        if len(df_info) != len(df_prediction):
            warnings.warn("The two datasets do not have same length, maybe there is some intern problems", Warning)
        result = df_info.copy()
        for key in [key for key in df_prediction.columns if key not in df_info.columns]:
            result[key]=df_prediction[key]
        result = pd.concat([result,df_eval],axis=1,join="inner")
        return result
