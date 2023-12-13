MODEL = 'CSRA'
OBSERVED_CLASS='car'
PREDICTION_PATH = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\prediction_CSRA.csv"
OBSERVATION_PATH = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\DATA\\COCO_World_URLs\\Benchmark_data\\{}\\exploration\\{}_{}.xlsx".format(MODEL,MODEL,OBSERVED_CLASS)


MATCH_CLS = ['dog', 'person', 'book', 'handbag', 'bird', 'train', 'cake', 'tv', 'airplane', 'cow', 'car', 'boat', 'cat', 'bench', 'vase', 'motorcycle', 'sheep', 'horse', 'bicycle', 'cup', 'bottle', 'oven', 'chair', 'bus', 'couch', 'bed', 'toaster', 'tie', 'truck', 'umbrella', 'bowl', 'scissors', 'spoon', 'backpack', 'clock', 'sink', 'surfboard', 'banana', 'keyboard', 'fork', 'bear', 'apple', 'laptop', 'microwave', 'elephant', 'suitcase', 'zebra', 'sandwich', 'carrot', 'kite', 'mouse', 'frisbee', 'giraffe', 'toothbrush', 'orange', 'skateboard', 'toilet', 'knife', 'pizza']

GRADCAM=True
CONF_NAME = "gradcam_coco_resnet101"
GRADCAM_ONLY = False
