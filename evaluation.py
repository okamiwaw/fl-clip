import os
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.client import Client
from medclip.multi_fusion import MLPFusion_Mdoel
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


def get_valid_dataloader(client, data_type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    cls_prompts = generate_chexpert_class_prompts(n=10)
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    client=client,
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
select_model = MLPFusion_Mdoel(num_classes=constants.SELECT_NUM)
select_dict = torch.load('./outputs/models/best/select_model.pth', map_location=torch.device('cuda:0'))
select_model.load_state_dict(select_dict)
select_model.to("cuda:0")
global_dict = torch.load('./outputs/models/best_model/global_model.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict)
thd = constants.THRESHOLD
client_ids = [ "client_1", "client_2","client_3", "client_4"]
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
    val_data = get_valid_dataloader(client_id, "test")
    pred_label = []
    label_list = []
    tasks = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    cnt = 0
    for i, batch_data in enumerate(val_data):
        pixel = batch_data["pixel_value"].to("cuda:0")
        logits = []
        report_ids = batch_data["report"]["input_ids"].to("cuda:0")
        report_mask = batch_data["report"]["attention_mask"].to("cuda:0")
        for task in tasks:
            input_ids = batch_data["prompt_input"][task]["input_ids"].view(1, -1).to("cuda:0")
            attention_mask = batch_data["prompt_input"][task]["attention_mask"].view(1, -1).to("cuda:0")
            outputs = select_model(pixel, report_ids, report_mask).cpu().detach().numpy()
            max_index = np.argmax(outputs)
            person_model = person_models[client_ids[max_index]]
            if np.max(outputs) <= thd:
                person_model = global_model
            inputs={"input_ids":input_ids,
                    "attention_mask":attention_mask,
                    "pixel_values":pixel}
            medclip_outputs = person_model(**inputs)
            logit = medclip_outputs['logits'].cpu().detach().numpy()
            logits.append(logit)
        pred = np.argmax(logits)
        pred_label.append(pred)
        label_list.append(batch_data['label'])
    labels = torch.cat(label_list).cpu().detach().numpy()
    labels = np.argmax(labels, axis=1)
    labels = labels.tolist()
    acc = sum(x == y for x, y in zip(pred_label, labels)) / len(labels)
    print(f'personal model in {client_id} its acc is {acc}')

def eval_global(client_id):
    val_data = get_valid_dataloader(client_id, "test")
    pred_label = []
    label_list = []
    tasks = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    cnt = 0
    for i, batch_data in enumerate(val_data):
        pixel = batch_data["pixel_value"].to("cuda:0")
        logits = []
        for task in tasks:
            input_ids = batch_data["prompt_input"][task]["input_ids"].view(1, -1).to("cuda:0")
            attention_mask = batch_data["prompt_input"][task]["attention_mask"].view(1, -1).to("cuda:0")
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "pixel_values": pixel}
            medclip_outputs = global_model(**inputs)
            logit = medclip_outputs['logits'].cpu().detach().numpy()
            logits.append(logit)
        pred = np.argmax(logits)
        pred_label.append(pred)
        label_list.append(batch_data['label'])
    labels = torch.cat(label_list).cpu().detach().numpy()
    labels = np.argmax(labels, axis=1)
    labels = labels.tolist()
    acc = sum(x == y for x, y in zip(pred_label, labels)) / len(labels)
    print(f'global model in {client_id} its acc is {acc}')

for i in range(10):
    print(f"round: {i}")
    for client_id in client_ids:
            eval_personal(client_id)
            eval_global(client_id)