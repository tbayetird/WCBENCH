import pandas as pd
import os

classes_report_path = ""
classes_dir = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\DollarStreetDataset\\dataset_dollarstreet\\Benchmark_data\\CSRA\\exploration\\"
# classes_dir = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\CSRA\\exploration\\"

def get_all_paths(classes_dir):
    classes_tab = []
    for root,dirs,files in os.walk(classes_dir):
        for file in files :
            if '.xlsx' in file :
                classes_tab.append(os.path.join(root,file))
    return classes_tab

def get_fusion(classes_dir):
    paths = get_all_paths(classes_dir)
    # context and bad data étaient inversées dans l'outil, on les ré-inverse ici pour équilibrer.
    columns=['Bad annotation', 'Bad prediction', 'Lack Prediction', 'Context', 'Bad data', 'Other', 'Good']
    all_occ = pd.DataFrame(None, columns=columns)
    all_not_occ = pd.DataFrame(None,columns=columns)
    for path in paths :
        print(path)
        class_tab = pd.read_excel(path,engine="openpyxl",sheet_name="stats")
        class_occ_df = class_tab.iloc[1:,1:8].dropna()
        class_occ_df.columns=columns
        # print(class_occ_df)

        class_not_occ_df = class_tab.iloc[1:,8:].dropna()
        class_not_occ_df.columns = columns
        # print(class_not_occ_df)

        if all_occ['Bad annotation'].empty:
            all_occ = class_occ_df
            all_not_occ = class_not_occ_df
        else :
            for key in all_occ:
                all_occ[key]+=class_occ_df[key]
                all_not_occ[key]+=class_not_occ_df[key]

    return all_occ, all_not_occ

def get_acc(df):
    # Accuracy is total good predictions divided on total good and bad predictions
    total_good = df['Bad annotation'] + df['Good']
    total_bad = df['Bad prediction'] + df['Lack Prediction']
    return float(total_good / (total_bad + total_good))

def get_precision(df):
    # TP = True positive, FP = False positive , PP = Predicted positive = TP + FP
    # precision = TP / TP + FP  = Bad ann + Good / Bad ann + good + Bad pred
    TP = df['Bad annotation'] + df['Good']
    PP = TP + df['Bad prediction']
    return float(TP / PP)

def get_recall(df):
    # recall = TP / TP + FN
    TP = df['Bad annotation'] + df['Good']
    FN = df['Lack Prediction']
    return float(TP /(TP + FN))

def get_F1(df):
    prec = get_precision(df)
    rec = get_recall(df)
    return 2* (prec * rec)/(prec + rec)

def get_bad_data_rate(df):
    bad_data_count = df['Bad data']
    all_count = sum([int(k) for k in [df[key] for key in df] ])
    return float(bad_data_count / all_count)


def report_df(df,name =""):
    print(' --- Starting report for dataframe {} --- \n'.format(name))
    print(' ## Accuracy : {}'.format(get_acc(df)))
    print(' ## Recall : {}'.format(get_recall(df)))
    print(' ## Precision : {}'.format(get_precision(df)))
    print(' ## F1 score : {}'.format(get_F1(df)))
    print(' ## Bad data rate : {}'.format(get_bad_data_rate(df)))
    print(' --- End report for dataframe {} ---\n'.format(name))

def report():
    all_occ, all_not_occ = get_fusion(classes_dir)
    report_df(all_occ,'Occidental data')
    report_df(all_not_occ,' Non occidental data')

def report_transport_classes():
    columns=['Bad annotation', 'Bad prediction', 'Lack Prediction', 'Context', 'Bad data', 'Other', 'Good']
    transport_classes = ['bicycle','car','motorcycle','airplane','bus','train','truck','boat']
    transport_paths = [classes_dir+"CSRA_"+elem+'.xlsx' for elem in transport_classes]
    #hahaha, this takes also the carrot csv. funny.
    for i,path in enumerate(transport_paths):
        print(" Computing {} class ".format(transport_classes[i]))
        class_tab = pd.read_excel(path,engine="openpyxl",sheet_name="stats")

        class_occ_df = class_tab.iloc[1:,1:8].dropna()
        class_occ_df.columns=columns

        class_not_occ_df = class_tab.iloc[1:,8:].dropna()
        class_not_occ_df.columns = columns

        occ_acc = get_acc(class_occ_df)
        not_occ_acc = get_acc(class_not_occ_df)
        overall_acc = (len(class_occ_df)*occ_acc + len(class_not_occ_df)*not_occ_acc)/(len(class_not_occ_df)+len(class_occ_df))
        print(" ## Accuracy OCC : {}".format(occ_acc))
        print(" ## Accuracy NOT OCC : {}".format(not_occ_acc))
        print(" ## Accuracy overall : {}".format(overall_acc))

# report()
report_transport_classes()
