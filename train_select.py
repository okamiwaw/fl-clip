import os
import threading
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.client import Client
from medclip.multi_fusion import MLPFusion_Mdoel,CAFusion_Mdoel
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


def get_valid_dataloader(client , data_type):
    dataset_path = constants.DATASET_PATH
    datalist_path = constants.DATALIST_PATH
    val_data = ZeroShotImageDataset(class_names=constants.CHEXPERT_COMPETITION_TASKS,
                                    dataset_path=dataset_path,
                                    datalist_path=datalist_path,
                                    client=client,
                                    data_type=data_type)
    val_collate_fn = ZeroShotImageCollator(mode='multiclass')
    val_dataloader = DataLoader(val_data,
                                batch_size=50,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=2,
                                )
    return val_dataloader


def log_metric(r, type, acc):
    log_file = './outputs/log/log_select.txt'
    folder_path = os.path.dirname(log_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(log_file, 'a') as f:
        f.write(f'Round:{r} {type} :ACC: {acc:.4f}\n')


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
        # select_model = MLPFusion_Mdoel(
        #     num_classes=client_nums
        # )
        select_model = CAFusion_Mdoel(
            num_classes=client_nums
        )
        self.server = Server(global_model=global_model,
                             select_model=select_model,
                             client_ids=self.client_ids)

    def train(self):
        server = self.server
        select_method = 'ca'
        writer = SummaryWriter(f'outputs/log/fl-select_{select_method}')
        select_acc = 0
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
                                select_method=select_method,
                                select_dict=server.select_model.state_dict(),
                                select_label=clients_label[client_id]
                                )
                client.select_train()
                diff_select = client.compute_diff(server.select_model, "select")
                server.receive(client_id=client_id,
                               model=diff_select,
                               model_type="select_model"
                               )
            server.aggregate()
            select_model = server.select_model.to("cuda:0")
            select_model.eval()
            with torch.no_grad():
                metric = 0
                for i, batch_data in enumerate(val_global):
                    # input and expected output
                    pixel = batch_data["pixel_values"].to("cuda:0")
                    input_ids = batch_data["input_ids"].to("cuda:0")
                    attention_mask = batch_data["attention_mask"].to("cuda:0")
                    client_ids = batch_data["clients"]
                    label_mapping = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                    select_labels = [label_mapping[client_id] for client_id in client_ids]
                    labels = np.ones((pixel.shape[0], 1)) * select_labels
                    outputs = select_model(pixel, input_ids, attention_mask)
                    labels = labels.argmax(1)
                    pred = outputs.argmax(1).cpu().numpy()
                    acc = (pred == labels).mean()
                    metric += acc
                metric /= len(val_global)
                writer.add_scalar(f'select_{select_method}_acc', metric, r)
                log_metric(r, 'select', metric)
                print(f"select model acc is {metric}")
            if metric > select_acc:
                select_acc = metric
                print(f'metric:{select_acc}')
                save_dir = f'outputs/models/best'
                os.makedirs(save_dir, exist_ok=True)
                select_path = os.path.join(save_dir, f'select_model_{select_method}.pth')
                torch.save(server.select_model.state_dict(), select_path)
            select_model.to("cpu")


def main():
    runner = Runner()
    runner.config()
    runner.train()


if __name__ == "__main__":
    main()
