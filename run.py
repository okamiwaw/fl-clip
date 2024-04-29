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
from medclip.vgg import vgg11


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
        global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        select_model = vgg11(
            num_classes=constants.SELECT_NUM
        )
        state_dict = torch.load('./outputs/models/best/select_model.pth')
        select_model.load_state_dict(state_dict)
        log_file = constants.LOGFILE
        self.server = Server(global_model=global_model,
                             select_model=select_model,
                             client_ids=self.client_ids,
                             log_file=log_file)

    def train(self):
        server = self.server
        for r in range(self.rounds):
            print(f"round {r} / {self.rounds} is beginning!")
            for client_id in self.client_ids:
                print(f"{client_id} is starting training!")
                log_file = constants.LOGFILE
                train_dataloader = get_train_dataloader(client_id)
                val_person = get_valid_dataloader(client_id)
                val_global = get_valid_dataloader('global')
                clients_label = constants.CLIENTS_LABEL
                client = Client(client_id=client_id,
                                train_dataloader=train_dataloader,
                                val_person=val_person,
                                val_global=val_global,
                                round=r,
                                log_file=log_file,
                                local_dict=server.global_model.state_dict(),
                                person_dict=server.person_models[client_id].state_dict(),
                                select_dict=server.select_model.state_dict(),
                                select_label=clients_label[client_id]
                                )
                client.save_best_model("local")
                client.save_best_model("person")
                p1 = mp.Process(target=client.person_train)
                p2 = mp.Process(target=client.local_train)
                p1.start()
                p2.start()
                p1.join()
                p2.join()
                diff_local = client.compute_diff(server.global_model, "global")
                server.receive(client_id=client_id,
                               global_dict=diff_local,
                               person_model=client.person_model
                               )
            val_global = get_valid_dataloader('global')
            server.aggregate()
            server.validate(val_global)


def main():
    runner = Runner()
    runner.config()
    runner.train()


if __name__ == "__main__":
    main()
