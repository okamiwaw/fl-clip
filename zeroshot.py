import os
import numpy as np
import torch
from PIL import Image

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier, MedCLIPProcessor
import pandas as pd

from medclip.prompts import process_class_prompts, generate_rsna_class_prompts

global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
global_dict = torch.load('./outputs/models/best_model/global_model.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict)
clf = PromptClassifier(global_model, ensemble=True)
cls_prompts = process_class_prompts(generate_rsna_class_prompts(n=10))
processor = MedCLIPProcessor()

RSNA_path = 'data/data_list/rsna.csv'
COVID_path = 'data/data_list/covid.csv'
RSNA_data = 'data/data_set/RSNA'
COVID_data = 'data/data_set/COVID'
rsna = pd.read_csv(RSNA_path)
covid = pd.read_csv(COVID_path)
labels = []
predict = []
for i, row in rsna.iterrows():
    path = RSNA_data + '/' + row['imgpath'] + '.jpg'
    if row['label'] == 'Normal':
        labels.append(1)
    else:
        labels.append(0)
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    inputs['prompt_inputs'] = cls_prompts
    output = clf(**inputs)
    max_index = torch.argmax(output['logits']).item()
    predict.append(max_index)
print('a')


