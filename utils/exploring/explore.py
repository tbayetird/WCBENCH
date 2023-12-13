import os,sys
import pandas as pd
import PIL.Image, PIL.ImageDraw, PIL.ImageTk
from tkinter import *

sys.path.insert(0,"C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\CSRA")
from gradCAM_visualiser import GradCAM_visualiser as gc

def from_ann_to_list(ann):
    if ann == '[]':
        return []
    ann_out = ann[1:-1].replace('\'','').split(', ')
    return ann_out

class GC_args :

    def __init__(self,conf_name):
        self.config_name = conf_name

class ImageViewer:
    def __init__(self, master, conf):
        self.master = master
        self.master.title("Data Analysis Tool")

        self.image_label = Label(self.master)
        self.image_label.pack()

        self.current_image_index = 0
        self.match_cls = conf.MATCH_CLS
        print(len(conf.MATCH_CLS))
        self.observed_class = conf.OBSERVED_CLASS
        self.observation_path=conf.OBSERVATION_PATH
        self.load_data(conf)
        self.load_image()
        self.create_navigation_buttons()
        if conf.GRADCAM == True :
            self.load_gradcam_model(conf)

    def load_data(self,conf):
        if os.path.exists(conf.OBSERVATION_PATH):
            print('loading existing data')
            data_df = pd.read_excel(conf.OBSERVATION_PATH,engine="openpyxl",sheet_name="all")
        else :
            data_df = pd.read_csv(conf.PREDICTION_PATH)
            data_df['observations']=None
        bool_ann = data_df['annotation'].apply(lambda x : self.observed_class in from_ann_to_list(x))
        bool_pred = data_df['prediction'].apply(lambda x : self.observed_class in from_ann_to_list(x))
        class_data_df=data_df.loc[bool_ann+bool_pred]

        # add a new column to stock observations results
        self.data_df = class_data_df.reset_index(drop=True)
        print(self.data_df)

    def load_gradcam_model(self,conf):
        self.gradcam_model = gc(GC_args(conf.CONF_NAME))

    '''
    Takes an image and a row and add information such as source, preds and
    annotations on the image
    '''
    def add_info(self,im,data_row):
        im = im.resize((500,500))
        im_ann = data_row['annotation']
        im_pred = data_row['prediction']
        im_zone = data_row['imgPath'].split('\\')[-2].replace('_400','')

        # add those info on the image
        draw = PIL.ImageDraw.Draw(im)
        ann_color_bool=self.observed_class in from_ann_to_list(im_ann)
        pred_color_bool=self.observed_class in from_ann_to_list(im_pred)
        color_ann = (0,255,0) if ann_color_bool else (255,0,0)
        color_pred = (0,255,0) if pred_color_bool else (255,0,0)
        color_zone = (255,255,255) if data_row['occident'] else (0,0,0)

        draw.text((0, 0),im_zone,color_zone)
        draw.text((0, 20),'observed_class : {}'.format(self.observed_class),(0,0,0))
        draw.text((0, 40),im_ann,color_ann)
        draw.text((0, 60),im_pred,color_pred)

        return im

    '''
    Takes an image and returns the gradcam of the image.
    '''
    def get_gradcam(self,image):
        gc_image = self.gradcam_model.get_gradcam_image(image,self.observed_class)
        return gc_image


    def load_gradcam(self):
        print('loading gradcam for image with index {}'.format(self.current_image_index))
        if 0 <= self.current_image_index < len(self.data_df):
            data_row = self.data_df.iloc[self.current_image_index]
            image_path = data_row['imgPath']
            image = PIL.Image.open(image_path)
            image = self.get_gradcam(image)
            image = self.add_info(image,data_row)
            image.thumbnail((800, 800))  # Resize the image if necessary
            photo = PIL.ImageTk.PhotoImage(image=image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def load_image(self):
        print('loading image with index {}'.format(self.current_image_index))
        if 0 <= self.current_image_index < len(self.data_df):
            data_row = self.data_df.iloc[self.current_image_index]
            image_path = data_row['imgPath']
            image = PIL.Image.open(image_path)
            image = self.add_info(image,data_row)
            image.thumbnail((800, 800))  # Resize the image if necessary
            photo = PIL.ImageTk.PhotoImage(image=image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def create_navigation_buttons(self):
        prev_button = Button(self.master, text="Previous", command=self.show_previous_image)
        prev_button.pack(side=LEFT)
        good_button = Button(self.master, text="Good Pred", command=self.add_good_pred)
        good_button.pack(side=LEFT)

        next_button = Button(self.master, text="Next", command=self.show_next_image)
        next_button.pack(side=RIGHT)
        Bad_ann_button = Button(self.master, text="Bad_ann", command=self.add_bad_ann)
        Bad_ann_button.pack(side=RIGHT)
        Bad_pred_button = Button(self.master, text="Bad_pred", command=self.add_bad_pred)
        Bad_pred_button.pack(side=RIGHT)
        lack_pred_button = Button(self.master, text="Lack_pred", command=self.add_lack_pred)
        lack_pred_button.pack(side=RIGHT)
        Context_switchbutton = Button(self.master, text="Context_switch", command=self.add_context)
        Context_switchbutton.pack(side=RIGHT)
        Bad_data_button = Button(self.master, text="Bad_data", command=self.add_bad_data)
        Bad_data_button.pack(side=RIGHT)
        gradcam_button = Button(self.master, text="gradcam", command=self.show_gradcam)
        gradcam_button.pack(side=RIGHT)


    def show_previous_image(self):
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.data_df) - 1
        self.load_image()

    def show_next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.data_df):
            self.current_image_index = 0
        self.load_image()

    def add_good_pred(self):
        self.data_df.loc[self.current_image_index,'observations']='good_prediction'
        self.show_next_image()

    def add_bad_ann(self):
        self.data_df.loc[self.current_image_index,'observations']='bad_annotation'
        self.show_next_image()

    def add_bad_pred(self):
        self.data_df.loc[self.current_image_index,'observations']='bad_prediction'
        self.show_next_image()

    def add_lack_pred(self):
        self.data_df.loc[self.current_image_index,'observations']='lack_prediction'
        self.show_next_image()

    def add_bad_data(self):
        self.data_df.loc[self.current_image_index,'observations']='bad_data'
        self.show_next_image()

    def add_context(self):
        self.data_df.loc[self.current_image_index,'observations']='context'
        self.show_next_image()

    def show_gradcam(self):
        self.load_gradcam()


    def save(self):
        self.data_df.to_csv(self.observation_path)
        pass


class Explorer :

    def __init__(self,conf):
        self.root = Tk()
        self.app = ImageViewer(self.root,conf)
        self.observation_path = conf.OBSERVATION_PATH

    def run(self):
        self.root.mainloop()
        self.create_stats_visualisations(self.app.data_df)

    def create_stats_visualisations(self,data_df):
        occ_df = self.generate_stats(data_df.loc[data_df['occident'].apply(lambda x : x)])
        occ_df = occ_df.rename(columns={'occ or not occ': 'occident'}, level=0)
        not_occ_df = self.generate_stats(data_df.loc[data_df['occident'].apply(lambda x : not x)])
        not_occ_df = not_occ_df.rename(columns={'occ or not occ': 'not occident'}, level=0)

        result = pd.concat([occ_df, not_occ_df], axis=1)
        #if needed, replace csv by excel.
        excel_name = self.observation_path.replace('.csv','_excel.xlsx')

        if not os.path.exists(os.path.dirname(excel_name)):
            os.mkdir(os.path.dirname(excel_name))

        with pd.ExcelWriter(excel_name) as writer:
            data_df.to_excel(writer, sheet_name='all')
            result.to_excel(writer, sheet_name='stats')

    def generate_stats(self,data_df):
        bad_ann_count = len(data_df.loc[data_df['observations']=='bad_annotation'])
        bad_pred_count = len(data_df.loc[data_df['observations']=='bad_prediction'])
        lack_pred_count = len(data_df.loc[data_df['observations']=='lack_prediction'])
        bad_data_count = len(data_df.loc[data_df['observations']=='bad_data'])
        context_count = len(data_df.loc[data_df['observations']=='context'])
        other_count = len(data_df.loc[data_df['observations']=='other'])
        good_count_1 = len(data_df.loc[data_df['observations'].apply(lambda x : x is None)])
        good_count_2 = len(data_df.loc[data_df['observations']=='good_prediction'])
        good_count = good_count_1 + good_count_2
        stats_df = pd.DataFrame([[bad_ann_count,bad_pred_count,lack_pred_count,bad_data_count,context_count,other_count,good_count]],
                    columns=pd.MultiIndex.from_product([['occ or not occ'],
                                                        ['Bad annotation','Bad prediction','Lack Prediction','Bad data', 'Context','Other','Good']]))
        return stats_df
