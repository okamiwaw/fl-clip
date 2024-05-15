import os
import threading
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
                                  batch_size=50,
                                  collate_fn=train_collate_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=8,
                                  )
    return train_dataloader


def get_valid_dataloader( data_type):
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
                                batch_size=50,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                )
    return val_dataloader

class Runner:
    def __init__(self):
        # set the initial environment
        self.client_ids = None
        self.rounds = None
        self.clients = None
        self.server = None
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONASHSEED'] = str(seed)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        mp.set_start_method('spawn')


    def config(self):
        self.client_ids = constants.CLIENT_IDS
        self.rounds = constants.ROUNDS
        client_nums = constants.SELECT_NUM
        global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        select_model_image = vgg11(
            num_classes=client_nums
        )
        select_model_text = Bert_Classifier(
            num_classes=client_nums
        )
        self.server = Server(global_model=global_model,
                             select_model_image=select_model_image,
                             select_model_text=select_model_text,
                             client_ids=self.client_ids)

    def train(self):
        server = self.server
        select_image_acc = 0
        select_text_acc = 0
        for r in range(200):
            print(f"round {r} / 200 is beginning!")
            val_global = get_train_dataloader('global')
            for client_id in self.client_ids:
                print(f"{client_id} is starting training!")
                log_file = constants.LOGFILE
                train_dataloader = get_train_dataloader(client_id)
                val_person = get_train_dataloader(client_id)
                clients_label = constants.CLIENTS_LABEL
                client = Client(client_id=client_id,
                                train_dataloader=train_dataloader,
                                val_person=val_person,
                                val_global=val_global,
                                round=r,
                                log_file=log_file,
                                local_dict=server.global_model.state_dict(),
                                person_dict=server.person_models[client_id].state_dict(),
                                select_dict_image=server.select_model_image.state_dict(),
                                select_dict_text=server.select_model_text.state_dict(),
                                select_label=clients_label[client_id]
                                )
                client.select_train()
                diff_select_image = client.compute_diff(server.select_model_image, "select_image")
                diff_select_text = client.compute_diff(server.select_model_text, "select_text")
                server.receive(client_id=client_id,
                               model=diff_select_image,
                               model_type="select_image"
                               )
                server.receive(client_id=client_id,
                               model=diff_select_text,
                               model_type="select_text"
                               )
            server.aggregate()
            select_model_image = server.select_model_image.to("cuda:0")
            select_model_text = server.select_model_text.to("cuda:0")
            select_model_image.eval()
            select_model_text.eval()
            with torch.no_grad():
                metric1 = 0
                metric2 = 0
                for i, batch_data in enumerate(val_global):
                    # input and expected output
                    images = batch_data["pixel_values"].to("cuda:0")
                    client_ids = batch_data["clients"]
                    label_mapping = [[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]]
                    select_labels = [label_mapping[client_id] for client_id in client_ids]
                    labels = np.ones((images.shape[0], 1)) * select_labels
                    outputs_image = select_model_image(images)
                    input_ids = batch_data["input_ids"].to("cuda:0")
                    attention_mask = batch_data["attention_mask"].to("cuda:0")
                    outputs_text = select_model_text(input_ids, attention_mask)
                    labels = labels.argmax(1)
                    pred1 = outputs_image.argmax(1).cpu().numpy()
                    pred2 = outputs_text.argmax(1).cpu().numpy()
                    acc1 = (pred1 == labels).mean()
                    acc2 = (pred2 == labels).mean()
                    metric1 += acc1
                    metric2 += acc2
                metric1 /= len(val_global)
                metric2 /= len(val_global)
                print(f"select model_image acc is {metric1}")
                print(f"select model_text acc is {metric2}")
            if metric1 > select_image_acc:
                select_image_acc = metric1
                save_dir = f'outputs/models/best'
                os.makedirs(save_dir, exist_ok=True)
                select_path = os.path.join(save_dir, "outputs/models/best/select_model_image.pth")
                torch.save(server.select_model_image.state_dict(), select_path)
            if metric2 > select_text_acc:
                select_text_acc = metric2
                save_dir = f'outputs/models/best'
                os.makedirs(save_dir, exist_ok=True)
                select_path = os.path.join(save_dir, "outputs/models/best/select_model_text.pth")
                torch.save(server.select_model_text.state_dict(), select_path)
            select_model_image.to("cpu")
            select_model_text.to("cpu")


def main():
    runner = Runner()
    runner.config()
    runner.train()


if __name__ == "__main__":
    main()
