import os
import numpy as np
import torch
from PIL import Image

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier, MedCLIPProcessor
import pandas as pd

from medclip.prompts import process_class_prompts, generate_rsna_class_prompts, generate_covid_class_prompts

global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
# global_model.from_pretrained()

global_dict = torch.load('./outputs/models/best_model/global_model_fed_moon.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict)
clf = PromptClassifier(global_model, ensemble=True)
cls_prompts = process_class_prompts(generate_rsna_class_prompts(5))

processor = MedCLIPProcessor()
RSNA_path = 'data/data_list/rsna.csv'
COVID_path = 'data/data_list/covid.csv'
RSNA_data = 'data/data_set/RSNA'
COVID_data = 'data/data_set/COVID'
rsna = pd.read_csv(RSNA_path)
covid = pd.read_csv(COVID_path)


# labels = []
# predict = []
# cls_prompts = process_class_prompts(generate_covid_class_prompts(n=10))
# for i, row in covid.iterrows():
#     path = COVID_data + '/' + row['imgpath']
#     if row['label'] == 'positive':
#         labels.append(0)
#     else:
#         labels.append(1)
#     image = Image.open(path)
#     inputs = processor(images=image, return_tensors="pt")
#     inputs['prompt_inputs'] = cls_prompts
#     output = clf(**inputs)
#     max_index = torch.argmax(output['logits']).item()
#     predict.append(max_index)
# acc = sum(x == y for x, y in zip(predict, labels)) / len(labels)
# print(acc)

labels = []
predicts = []
for i, row in rsna.iterrows():
    print(i)
    # if i < 500:
    #     continue
    path = RSNA_data + '/' + row['imgpath'] + '.jpg'
    if row['label'] == 1:
        labels.append(0)
    else:
        labels.append(1)
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    inputs['prompt_inputs'] = cls_prompts
    output = clf(**inputs)
    # softmax_tensor = torch.nn.functional.softmax( output['logits'], dim=1)
    if output['logits'][0,0]  < 0.5:
        predict = 1
    else:
        predict = 0
    predicts.append(predict)
acc = sum(x == y for x, y in zip(predicts, labels)) / len(labels)
print(acc)

