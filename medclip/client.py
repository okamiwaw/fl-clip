import copy

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from medclip import MedCLIPModel, MedCLIPVisionModelViT, constants, PromptClassifier
from medclip.evaluator import Evaluator
from medclip.losses import ImageTextContrastiveLoss
from medclip.vgg import vgg11




class Client:
    def __init__(self,
                 client_id=None,
                 train_dataloader=None,
                 val_dataloader=None,
                 deviceA='cpu',
                 deviceB='cpu',
                 round=0,
                 writer=None,
                 epochs=1,
                 local_dict=None,
                 person_dict=None,
                 select_dict=None,
                 select_label=None,
                 log_file=None
                 ):
        self.client_id = client_id
        self.round = round
        self.deviceA = deviceA
        self.deviceB = deviceB
        self.epochs = epochs
        self.log_file = log_file
        self.writer = writer
        self.select_label = select_label
        self.train_loader = train_dataloader
        self.valid_loader = val_dataloader
        self.local_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.deviceA)
        self.person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.deviceB)
        self.select_model = vgg11(
            num_classes=constants.SELECT_NUM
        ).to(self.deviceA)
        self.textvision_lr = constants.VIT_BERT_LEARNING_RATE
        self.weight_decay = constants.WEIGHT_DECAY
        self.select_lr = constants.SELECT_MODEL_LEARNING_RATE
        self.local_model.load_state_dict(copy.deepcopy(local_dict))
        self.person_model.load_state_dict(copy.deepcopy(person_dict))
        self.select_model.load_state_dict(copy.deepcopy(select_dict))
    def log_metric(self, client, task, acc):
        log_file = self.log_file
        with open(log_file, 'a') as f:
            f.write(f'Round: {self.round}, {client}-{task} :ACC: {acc:.4f}\n')
    def local_train(self):
        print("local model training starts")
        writer = self.writer
        loss_model = ImageTextContrastiveLoss(self.local_model).to(self.deviceA)
        param_optimizer = list(loss_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.textvision_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            writer.add_scalar(f'{self.client_id}-local', loss.item(), self.round * len(self.train_loader) + i)
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})

    def person_train(self):
        print("personal model training starts")
        writer = self.writer
        loss_model = ImageTextContrastiveLoss(self.person_model).to(self.deviceB)
        param_optimizer = list(loss_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.textvision_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss_return = loss_model(**batch_data)
                loss = loss_return['loss_value']
            writer.add_scalar(f'{self.client_id}-person', loss.item(), self.round * len(self.train_loader) + i)
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item()})


    def select_train(self):
        select_label = self.select_label
        writer = self.writer
        print("select model training starts")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.select_model.parameters(), lr=self.select_lr)
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
        scaler = GradScaler()
        for i, batch_data in progress_bar:
            optimizer.zero_grad()
            with autocast():
                inputs = batch_data["pixel_values"].to(self.deviceA)
                labels = np.ones((inputs.shape[0], 1)) * select_label
                labels = torch.tensor(labels).to(self.deviceA)
                outputs = self.select_model(inputs)
                loss = criterion(outputs, labels)
            writer.add_scalar(f'{self.client_id}-select', loss.item(), self.round * len(self.train_loader) + i)
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
            device=self.deviceA
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        print(f"local model acc is {metric}")
        self.log_metric(client=self.client_id, task='local', acc=metric)
        medclip_clf = PromptClassifier(self.person_model)
        evaluator = Evaluator(
            medclip_clf=medclip_clf,
            eval_dataloader=valid_loader,
            mode='multiclass',
            device=self.deviceB
        )
        scores = evaluator.evaluate()
        metric = scores['acc']
        print(f"personal model acc is {metric}")
        self.log_metric(client=self.client_id, task='person', acc=metric)
        self.select_model.eval()
        with torch.no_grad():
            metric = 0
            for i, batch_data in enumerate(valid_loader):
                # input and expected output
                images = batch_data["pixel_values"].to(self.deviceA)
                # generate label vector: image batch_size, same label
                labels = np.ones((images.shape[0], 1)) * select_label
                outputs = self.select_model(images)
                labels = labels.argmax(1)
                pred = outputs.argmax(1).cpu().numpy()
                acc = (pred == labels).mean()
                metric += acc
            metric /= len(valid_loader)
            print(f"select model acc is {metric}")
            self.log_metric(client=self.client_id, task='selector', acc=metric)
