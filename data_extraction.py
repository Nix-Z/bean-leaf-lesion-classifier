import pandas as pd
import os
from glob import glob

def load_data():
  # Create Dataframe for training data
  train_data = pd.DataFrame({"path":[], "label":[], "class_id":[]})
  train_file_path = 'train'
  folder_list = os.listdir(train_file_path)
  label_dict = {"healthy": 0, "angular_leaf_spot": 1, "bean_rust": 2,}

  for i, folder in enumerate(folder_list):
    img_path = os.path.join(train_file_path, folder)
    jpg_list = glob(img_path + '/*.jpg')
    for jpg in jpg_list:
      new_data = pd.DataFrame({"path": jpg, "label": folder, "class_id": label_dict[folder]}, index=[1])
      train_data = pd.concat([train_data, new_data], ignore_index=True)

  train_data[["path"]] = train_data[["path"]].astype(str)
  train_data[["label"]] = train_data[["label"]].astype(str)
  train_data[["class_id"]] = train_data[["class_id"]].astype(int)

load_data()
