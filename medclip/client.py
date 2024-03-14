import copy

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from medclip import MedCLIPModel, MedCLIPVisionModelViT, constants, PromptClassifier
from medclip.evaluator import Evaluator
from medclip.losses import ImageTextContrastiveLoss
from medclip.vgg import vgg11


class Client:
    def __init__(self,
                 client_id=None,
                 train_dataloader=None,
                 val_dataloader=None,
                 device='cpu',
                 rounds=0,
                 epochs=1,
                 local_dict=None,
                 person_dict=None,
                 select_dict=None,
                 select_label=None
                 ):
        self.client_id = client_id
        self.device = device
        self.epochs = epochs
        self.select_label = select_label
        self.train_loader = train_dataloader
        self.valid_loader = val_dataloader
        self.local_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.device)
        self.person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.device)
        self.select_model = vgg11(
            num_classes=constants.SELECT_NUM
        ).to(self.device)
        self.textvision_lr = constants.VIT_BERT_LEARNING_RATE
        self.select_lr = constants.SELECT_MODEL_LEARNING_RATE
        self.local_model.load_state_dict(copy.deepcopy(local_dict))
        self.person_model.load_state_dict(copy.deepcopy(person_dict))
        self.select_model.load_state_dict(copy.deepcopy(select_dict))

    def local_train(self):
        print("local model training starts")
        optimizer = optim.Adam(self.local_model.parameters(), lr=self.textvision_lr)
        loss_model = ImageTextContrastiveLoss(self.local_model).to(self.device)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})

    def person_train(self):
        print("personal model training starts")
        optimizer = optim.Adam(self.person_model.parameters(), lr=self.textvision_lr)
        loss_model = ImageTextContrastiveLoss(self.person_model).to(self.device)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})

    def select_train(self):
        select_label = self.select_label
        print("select model training starts")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.select_model.parameters(), lr=self.select_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                inputs = batch_data["pixel_values"].to(self.device)
                labels = np.ones((inputs.shape[0],1)) * select_label
                labels = torch.tensor(labels).to(self.device)
                outputs = self.select_model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})

    def compute_diff(self, model, model_type):
        global_dict = model.state_dict()
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

    def validate(self):
        valid_loader = self.valid_loader
        select_label = self.select_label
        medclip_clf = PromptClassifier(self.local_model)
        evaluator = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=valid_loader,
            mode='multiclass',
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        print(f"local model acc is {metric}")

        medclip_clf = PromptClassifier(self.person_model)
        evaluator = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=valid_loader,
            mode='multiclass',
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        print(f"personal model acc is {metric}")

        self.select_model.eval()
        with torch.no_grad():
            metric = 0
            for i, batch_data in enumerate(valid_loader):
                # input and expected output
                images = batch_data["pixel_values"].to(self.device)
                # generate label vector: image batch_size, same label
                labels = np.ones((images.shape[0], 1)) * select_label
                outputs = self.select_model(images)
                labels = labels.argmax(1)
                pred = outputs.argmax(1).cpu().numpy()
                acc = (pred == labels).mean()
                metric += acc
            metric /= len(valid_loader)
            print(f"select model acc is {metric}")
