import os
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.client import Client
from medclip.prompts import generate_chexpert_class_prompts
from medclip.sever import Server
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, ZeroShotImageDataset, \
    ZeroShotImageCollator
from medclip.select_model import vgg11
from medclip.select_model import Bert_Classifier


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum(axis=0)  # the sum is computed along the only axis (axis=0)


def get_valid_dataloader(data_type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    cls_prompts = generate_chexpert_class_prompts(n=10)
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    data_type=data_type)
    val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
                                           mode='multiclass')
    val_dataloader = DataLoader(val_data,
                                batch_size=1,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0,
                                )
    return val_dataloader


global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
select_model_image = vgg11(
    num_classes=constants.SELECT_NUM
)
select_model_text = Bert_Classifier(num_classes=constants.SELECT_NUM)
select_image_dict = torch.load('./outputs/models/best/select_model_image.pth', map_location=torch.device('cuda:0'))
select_text_dict = torch.load('./outputs/models/best/select_model_text.pth', map_location=torch.device('cuda:0'))
select_model_image.load_state_dict(select_image_dict)
select_model_text.load_state_dict(select_text_dict)
select_model_image.to("cuda:0")
select_model_text.to("cuda:0")
global_dict = torch.load('./outputs/models/best_model/global_model.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict)

thd = constants.THRESHOLD
client_ids = ["client_1", "client_2", "client_3", "client_4"]
person_models = {}
for client_id in client_ids:
    person_dict = torch.load(f'./outputs/models/best_model/person_model_{client_id}.pth',
                             map_location=torch.device('cuda:0'))
    # person_dict = torch.load(f'./outputs/models/best/{client_id}.pth',
    #                          map_location=torch.device('cuda:0'))
    person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    person_model.load_state_dict(person_dict)
    person_models[client_id] = person_model
for client_id in client_ids:
    person_models[client_id].to("cuda:0")


def eval_personal(client_id):
    val_data = get_valid_dataloader(client_id)
    pred_list = []
    label_list = []
    tasks = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    for i, batch_data in enumerate(val_data):
        image = batch_data["pixel_values"].to("cuda:0")
        outputs = select_model_image(image).cpu().detach().numpy()
        outputs = softmax(outputs)
        outputs2 = np.empty((1, 4))
        for task in tasks:
            input_ids = batch_data["prompt_inputs"][task]["input_ids"].to("cuda:0")
            attention_mask = batch_data["prompt_inputs"][task]["attention_mask"].to("cuda:0")
            if np.size(outputs2):
                outputs2 = select_model_text(input_ids, attention_mask).cpu().detach().numpy()
            else:
                outputs2 += select_model_text(input_ids, attention_mask).cpu().detach().numpy()
        outputs2 = outputs2.mean(axis=0).reshape(1, 4)
        outputs = (outputs * 2 + outputs2) / 3
        max_index = np.argmax(outputs)
        person_model = person_models[client_ids[max_index]]
        if np.max(outputs) <= thd:
            person_model = global_model
        person_model = person_models[client_id]
        medclip_clf = PromptClassifier(person_model)
        medclip_clf.eval()
        output = medclip_clf(**batch_data)
        pred = output['logits'].to("cuda:0")
        pred_list.append(pred)
        label_list.append(batch_data['labels'])
    pred_list = torch.cat(pred_list, 0)
    labels = torch.cat(label_list).cpu().detach().numpy()
    pred = pred_list.cpu().detach().numpy()
    pred_label = pred.argmax(1)
    acc = (pred_label == labels).mean()
    print(f'personal model in {client_id} its acc is {acc}')

def eval_global(client_id):
    val_data = get_valid_dataloader(client_id)
    global_model = person_models[client_id]
    pred_list = []
    label_list = []
    for i, batch_data in enumerate(val_data):
        medclip_clf = PromptClassifier(global_model)
        medclip_clf.eval()
        outputs = medclip_clf(**batch_data)
        pred = outputs['logits'].to("cuda:0")
        pred_list.append(pred)
        label_list.append(batch_data['labels'])
    pred_list = torch.cat(pred_list, 0)
    labels = torch.cat(label_list).cpu().detach().numpy()
    pred = pred_list.cpu().detach().numpy()
    pred_label = pred.argmax(1)
    acc = (pred_label == labels).mean()
    print(f'global model in {client_id} its acc is {acc}')

for client_id in client_ids:
    eval_personal(client_id)
    eval_global(client_id)