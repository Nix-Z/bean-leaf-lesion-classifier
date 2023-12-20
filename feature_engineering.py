import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from datavisualization import visualize_data


def engineer_features():
  train_data = visualize_data()

  # Transform Data
  train_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # Create Custom Dataset
  class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
      self.data = dataframe
      self.transform = transform

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
      img_path = self.data.iloc[index]['path']
      img = Image.open(image_path).convert("RGB")
      transformed_img = self.transforms(img)
      class_id = self.df.iloc[index]['class_id']
      return transformed_img, class_id

  train_dataset = MyDataset(train_data, train_transforms)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  
  print(f"\nTrain Data: {len(train_data)}")
  train_data.to_csv('bean_leaf_lesion_data.csv', index=False)

  return train_dataset

engineer_features()

  
