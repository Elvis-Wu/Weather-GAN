from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
import os
import pandas as pd #資料處理
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

# 設定dataset路徑
project_path = r"D:\NTUT\Weather_AI\self-project\GAN\weather-GAN"
train_data_path = str(project_path + r"\train_data")
output_data_path = str(project_path + r"output_data")

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
class My_Dataset(Dataset) :
    def __init__(self, path, transform=None) :
        super().__init__()
        self.image_files = os.listdir(path)
        self.list_files = []
        for file in self.image_files :
            self.list_files.append(os.path.join(path, file))

    # Indicate the total size of the dataset
    def __len__(self) :
        return len(self.image_files)
    
    # 1. Read from file (using numpy.fromfile, PIL.Image.open)
    # 2. Preprocess the data (torchvision.Transform).
    # 3. Return the data (e.g. image and label)
    def __getitem__(self, index) :
        image = Image.open(self.list_files[index]).convert('RGB')
        if transform is None :
            return image
        else :
            image = transform(image)
            return image

#讀取圖片，一次四張
if __name__ == '__main__' :
    #path_image = r"D:\NTUT\Weather_AI\self-project\GAN\weather-GAN\train_data"
    path_image = train_data_path
    train_data = My_Dataset(path_image, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    for image in train_dataloader :
        print(image.shape)
        #4(張), 3(RGB三通道), 3600(長), 3600(寬)
        break

#im = Image.open(r"D:\NTUT\Weather_AI\self-project\GAN\weather-GAN\train_data\00_56_37.png")
#im.show()

#切割訓練集與測試集
train_set, test_set = train_test_split(image, test_size=0.2, random_state=42)
train_set = np.array(train_set)
test_set = np.array(test_set)

#儲存訓練集和測試集
if not os.path.exists("train_set") :
    os.makedirs("train_set")
    print("train_set dir is create.")
if not os.path.exists("test_set") :
    os.makedirs("test_set")
    print("test_set dir is create.")
    
np.save("train_set/train_set.npy", train_set)
print("Store train set.")
np.save("test_set/test_set.npy", test_set)
print("Store test set.")