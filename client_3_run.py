import os
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.client import Client
from medclip.evaluator import Evaluator
from medclip.prompts import generate_chexpert_class_prompts
from medclip.sever import Server
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, ZeroShotImageDataset, \
    ZeroShotImageCollator
def get_train_dataloader(client_id):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])],
    )
    train_data = ImageTextContrastiveDataset(datalist_path=datalist_path, dataset_path=dataset_path,
                                             imgtransform=transform, client_id=client_id)
    train_collate_fn = ImageTextContrastiveCollator()
    train_dataloader = DataLoader(train_data,
                                  batch_size=48,
                                  collate_fn=train_collate_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=2,
                                  )
    return train_dataloader
def get_valid_dataloader(client_id, data_type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    cls_prompts = generate_chexpert_class_prompts(n=10)
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    data_type=data_type,
                                    client=client_id)
    val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
                                           mode='multiclass')
    val_dataloader = DataLoader(val_data,
                                batch_size=50,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=2,
                                )
    return val_dataloader

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
client_id = "client_3"
best_acc = 0
log_file = constants.LOGFILE
train_dataloader = get_train_dataloader(client_id, )
val_person = get_valid_dataloader(client_id, "valid")
global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
client = Client(client_id=client_id,
                train_dataloader=train_dataloader,
                val_person=val_person,
                log_file=log_file,
                local_dict=global_model.state_dict(),
                )
for r in range(100):
    print(f"{r} round starts")
    client.local_train()
    medclip_clf = PromptClassifier(client.local_model)
    evaluator = Evaluator(
        medclip_clf=medclip_clf,
        eval_dataloader=val_person,
        mode='multiclass',
    )
    scores = evaluator.evaluate()
    metric = scores['acc']
    print(f"local model acc is {metric}")
    log_file = "./outputs/log/client_2.txt"
    folder_path = os.path.dirname(log_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(log_file, 'a') as f:
        f.write(f'Round: {round}, {client_id} :ACC: {metric:.4f}\n')
    if metric > best_acc:
        best_acc = metric
        save_path = f'./outputs/models/best/{client_id}.pth'
        torch.save(client.local_model.state_dict(), save_path)

