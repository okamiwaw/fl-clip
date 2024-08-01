from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants
from medclip.dataset import ZeroShotImageDataset, ZeroShotImageCollator, ImageTextContrastiveDataset, \
    ImageTextContrastiveCollator
from medclip.prompt_net import PromptLearner, PromptTranslator


def get_train_dataloader(client_id, bs):
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
                                  batch_size=bs,
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
                                batch_size=1,
                                collate_fn=val_collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=2,
                                )
    return val_dataloader

client_ids = constants.CLIENT_IDS
if __name__ == '__main__':
    for client_id in client_ids:
        model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
        train_dataloader = get_valid_dataloader(client_id, 'test')
        model_dict = model.state_dict()
        prompt_learner = PromptLearner(model_dict=model_dict,
                                       lr= constants.VIT_BERT_LEARNING_RATE,
                                       device='cuda:0',
                                       weight_decay = constants.WEIGHT_DECAY,
                                       client_id=client_id,
                                       train_loader=train_dataloader
                                       )
        prompt_learner.train()
        prompt_learner.save(client_id=client_id)