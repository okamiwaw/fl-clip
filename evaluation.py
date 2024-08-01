import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.multi_fusion import MLPFusion_Mdoel
from medclip.multi_fusion import PromptLearner
from medclip.prompt_net import PromptTranslator
from medclip.prompts import generate_chexpert_class_prompts
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, ZeroShotImageDataset, \
    ZeroShotImageCollator

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum(axis=0)  # the sum is computed along the only axis (axis=0)

def log_metric(client_id, model, acc):
    log_file = './outputs/log/log_fl.txt'
    folder_path = os.path.dirname(log_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(log_file, 'a') as f:
        f.write(f'client model {model} in {client_id} data , its acc is {acc}\n')

def get_valid_dataloader(client, data_type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    client=client,
                                    data_type=data_type)
    val_collate_fn = ZeroShotImageCollator(mode='multiclass')
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
global_dict = torch.load('./outputs/models/best_model/global_model_fed_moon.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict)
thd = constants.THRESHOLD
client_ids = [ "client_1", "client_2","client_3", "client_4"]
person_models = {}
prompt_models = {}

for client_id in client_ids:
    person_dict = torch.load(f'./outputs/models/best_model/person_model_{client_id}.pth',
                             map_location=torch.device('cuda:0'))
    # person_dict = torch.load(f'./outputs/models/best/{client_id}.pth',
    #                          map_location=torch.device('cuda:0'))
    person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    person_model.load_state_dict(person_dict)
    person_models[client_id] = person_model
for client_id in client_ids:
    prompt_dict = torch.load(f'./outputs/models/best_model/{client_id}_promptNet.pth',
                             map_location=torch.device('cuda:0'))
    prompt_model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
    prompt_model.load_state_dict(prompt_dict)
    prompt_models[client_id] = prompt_model
for client_id in client_ids:
    person_models[client_id].to("cuda:0")
    prompt_models[client_id].to("cuda:0")


def eval_with_sm(client_id):
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
        print(i)
        pixel = batch_data["pixel_values"].to("cuda:0")
        logits = []
        # report_ids = batch_data["reports"].to("cuda:0")
        # report_mask = batch_data["reports"].to("cuda:0")
        for task in tasks:
            input_ids = batch_data["prompt_inputs"][task]["input_ids"].view(1, -1).to("cuda:0")
            attention_mask = batch_data["prompt_inputs"][task]["attention_mask"].view(1, -1).to("cuda:0")
            outputs = select_model(pixel, input_ids, attention_mask).cpu().detach().numpy()
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
        label_list.append(batch_data['labels'])
    labels = label_list
    acc = sum(x == y for x, y in zip(pred_label, labels)) / len(labels)
    print(f'personal model in {client_id} its acc is {acc}')
    return acc
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
        pixel = batch_data["pixel_values"].to("cuda:0")
        logits = []
        for task in tasks:
            input_ids = batch_data["prompt_inputs"][task]["input_ids"].view(1, -1).to("cuda:0")
            attention_mask = batch_data["prompt_inputs"][task]["attention_mask"].view(1, -1).to("cuda:0")
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "pixel_values": pixel}
            medclip_outputs = global_model(**inputs)
            logit = medclip_outputs['logits'].cpu().detach().numpy()
            logits.append(logit)
        pred = np.argmax(logits)
        pred_label.append(pred)
        label_list.append(batch_data['labels'].item())
    labels = label_list
    acc = sum(x == y for x, y in zip(pred_label, labels)) / len(labels)
    print(f'global model in {client_id} its acc is {acc}')


def eval_client(client_id):
    for cid in client_ids:
        model = person_models[client_id]
        val_data = get_valid_dataloader(cid, "test")
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
                medclip_outputs = model(**inputs)
                logit = medclip_outputs['logits'].cpu().detach().numpy()
                logits.append(logit)
            pred = np.argmax(logits)
            pred_label.append(pred)
            label_list.append(batch_data['labels'])
        labels = torch.cat(label_list).cpu().detach().numpy()
        labels = np.argmax(labels, axis=1)
        labels = labels.tolist()
        acc = sum(x == y for x, y in zip(pred_label, labels)) / len(labels)
        print(f'client model {client_id} in {cid} data , its acc is {acc}\n')
        log_metric(client_id=cid,model= client_id,acc=acc)

def eval_with_smp(client_id):
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
        pixel = batch_data["pixel_values"].to("cuda:0")
        logits = []
        reports = batch_data['reports']
        for task in tasks:
            tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE, local_files_only=True)
            tokenizer.model_max_length = 77
            cls_inputs = tokenizer(task, truncation=True, max_length=20, padding="max_length", return_tensors='pt')
            emb1 = global_model.encode_text(cls_inputs['input_ids'], cls_inputs['attention_mask']).to("cuda:0")
            emb1 = prompt_models['client_1'](emb1).reshape(1, 512)
            emb1 = F.pad(emb1, (0, 768 - 512))
            input_ids = batch_data["prompt_inputs"][task]["input_ids"].view(1, -1).to("cuda:0")
            attention_mask = batch_data["prompt_inputs"][task]["attention_mask"].view(1, -1).to("cuda:0")
            text_features = select_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            image_features = select_model.image_model(pixel)
            image_features = 0.1 * emb1 +  image_features
            combined_features = torch.cat((text_features, image_features), dim=1)
            x = torch.relu(select_model.fc1(combined_features))
            x = select_model.fc2(x)
            outputs = F.softmax(x, dim=1).cpu().detach().numpy()
            max_index = np.argmax(outputs)
            person_model = person_models[client_ids[max_index]]
            if np.max(outputs) <= 0.4:
                person_model = global_model
            inputs={"input_ids":input_ids,
                    "attention_mask":attention_mask,
                    "pixel_values":pixel}
            medclip_outputs = person_model(**inputs)
            logit = medclip_outputs['logits'].cpu().detach().numpy()
            logits.append(logit)
        pred = np.argmax(logits)
        pred_label.append(pred)
        label_list.append(batch_data['labels'])
    labels = label_list
    acc = sum(x == y for x, y in zip(pred_label, labels)) / len(labels)
    print(f'personal model in {client_id} its acc is {acc}')
    return acc
# for i in range(100):
#     for client_id in client_ids:
#         eval_client(client_id)
# for i in range(10):
for client_id in client_ids:
    # eval_personal(client_id)
    eval_with_smp(client_id)



