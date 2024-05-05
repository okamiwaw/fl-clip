from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator
from medclip.select_model import Bert_Classifier
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

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
                                  num_workers=0,
                                  )
    return train_dataloader

model = Bert_Classifier(num_classes=4).to("cuda:0")
train_dataloader = get_train_dataloader("client_1")
select_label = [1, 0 ,0 ,0]
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()
progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
scaler = GradScaler()
for i, batch_data in progress_bar:
    optimizer.zero_grad()
    with autocast():
        input_ids = batch_data["input_ids"].to("cuda:0")
        attention_mask = batch_data["attention_mask"].to("cuda:0")
        labels = np.ones((input_ids.shape[0], 1)) * select_label
        labels = torch.tensor(labels).to("cuda:0")
        outputs = model(input_ids,attention_mask)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    progress_bar.set_postfix({"loss": loss.item()})
