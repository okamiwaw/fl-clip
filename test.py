from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT
from medclip.dataset import ZeroShotImageDataset, ZeroShotImageCollator, ImageTextContrastiveDataset,ImageTextContrastiveCollator

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
                                         imgtransform=transform, client_id='valid_2')
train_collate_fn = ImageTextContrastiveCollator()
train_dataloader = DataLoader(train_data,
                              batch_size=50,
                              collate_fn=train_collate_fn,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0,
                              )
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.cuda()
for i, batch_data in enumerate(train_dataloader):
    outputs = model(batch_data)
    print('done')