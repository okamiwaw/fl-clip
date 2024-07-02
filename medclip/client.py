import copy
import os

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification

from medclip import MedCLIPModel, MedCLIPVisionModelViT, constants, PromptClassifier, MedCLIPProcessor
from medclip.evaluator import Evaluator
from medclip.losses import ImageTextContrastiveLoss
from medclip.multi_fusion import MLPFusion_Mdoel,CAFusion_Mdoel



class Client:
    def __init__(self,
                 client_id=None,
                 train_dataloader=None,
                 val_person=None,
                 val_global=None,
                 device='cpu',
                 round=0,
                 select_method="mlp",
                 local_dict=None,
                 person_dict=None,
                 select_dict=None,
                 select_label=None,
                 log_file=None
                 ):
        self.client_id = client_id
        self.round = round
        self.device = device
        self.log_file = log_file
        self.select_label = select_label
        self.train_loader = train_dataloader
        self.val_person = val_person
        self.val_global = val_global
        self.local_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to("cuda:0")
        self.person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to("cuda:0")
        if select_method == 'mlp':
            self.select_model = MLPFusion_Mdoel(num_classes=constants.SELECT_NUM).to("cuda:0")
        else:
            self.select_model = CAFusion_Mdoel(num_classes=constants.SELECT_NUM).to("cuda:0")
        self.textvision_lr = constants.VIT_BERT_LEARNING_RATE
        self.weight_decay = constants.WEIGHT_DECAY
        self.select_lr = constants.SELECT_MODEL_LEARNING_RATE
        if local_dict is not None:
            self.local_model.load_state_dict(copy.deepcopy(local_dict))
        if person_dict is not None:
            self.person_model.load_state_dict(copy.deepcopy(person_dict))
        if select_dict is not None:
            self.select_model.load_state_dict(copy.deepcopy(select_dict))


    def log_metric(self, client, task, acc):
        log_file = self.log_file
        folder_path = os.path.dirname(log_file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(log_file, 'a') as f:
            f.write(f'Round: {self.round}, {client}-{task} :ACC: {acc:.4f}\n')

    def local_train(self):
        print("local model training starts")
        loss_model = ImageTextContrastiveLoss(self.local_model).to("cuda:0")
        optimizer = optim.Adam(loss_model.parameters(), lr=self.textvision_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})

    def person_train(self):
        print("personal model training starts")
        loss_model = ImageTextContrastiveLoss(self.person_model).to("cuda:1")
        optimizer = optim.Adam(loss_model.parameters(), lr=self.textvision_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})
    def select_train(self):
        select_label = self.select_label
        print("select model training starts")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_image = optim.AdamW(self.select_model.parameters(), lr=self.select_lr,
                                      weight_decay=self.weight_decay)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer_image.zero_grad()
            with autocast():
                pixel = batch_data["pixel_values"].to("cuda:0")
                input_ids = batch_data["input_ids"].to("cuda:0")
                attention_mask = batch_data["attention_mask"].to("cuda:0")
                labels = np.ones((pixel.shape[0], 1)) * select_label
                labels = torch.tensor(labels).to("cuda:0")
                outputs = self.select_model(pixel=pixel,
                                            input_ids=input_ids,
                                            attention_mask=attention_mask)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer_image)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})
    def compute_diff(self, model, model_type):
        global_dict = model.to("cpu").state_dict()
        diff_dict = {}
        if model_type == "global":
            local_dict = self.local_model.to("cpu").state_dict()
            for key in global_dict.keys():
                diff_dict[key] = local_dict[key] - global_dict[key]
        elif model_type == "select":
            local_dict = self.select_model.to("cpu").state_dict()
            for key in global_dict.keys():
                diff_dict[key] = local_dict[key] - global_dict[key]
        return diff_dict

    def save_best_model(self, model_type):
        save_dir = f'outputs/models/best_model'
        os.makedirs(save_dir, exist_ok=True)
        if model_type == 'local':
            global_path = os.path.join(save_dir, "global_model.pth")
            torch.save(self.local_model.state_dict(), global_path)
        if model_type == 'person':
            model_path = os.path.join(save_dir, f"person_model_{self.client_id}.pth")
            torch.save(self.person_model.state_dict(), model_path)

    def validate_global(self,writer):
        valid_global = self.val_global
        medclip_clf = PromptClassifier(self.local_model)
        evaluator = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=valid_global,
            mode='multiclass',
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        if metric > constants.GLOBAL_ACC:
            self.save_best_model('local')
            constants.GLOBAL_ACC = metric
        print(f"global model acc is {metric}")
        writer.add_scalar(f'global-{self.client_id}/fl-train', metric, self.round)
        self.log_metric(self.client_id, "global", metric)
        self.log_metric(self.client_id, "global_best", constants.GLOBAL_ACC)

    def validate_person(self,writer):
        valid_person = self.val_person
        medclip_clf = PromptClassifier(self.person_model)
        evaluator = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=valid_person,
            mode='multiclass',
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        if metric > constants.CLIENT_ACC[self.client_id]:
            self.save_best_model('person')
            constants.CLIENT_ACC[self.client_id] = metric
        print(f"personal model acc is {metric}")
        writer.add_scalar(f'personal-{self.client_id}/fl-train', metric, self.round)
        self.log_metric(self.client_id, "person_best", constants.CLIENT_ACC[self.client_id])