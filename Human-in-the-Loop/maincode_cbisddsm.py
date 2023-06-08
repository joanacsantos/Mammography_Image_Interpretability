import os
import time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, jaccard_score
import sys
import shutil
import csv
import pathlib
from pathlib import Path
import logging
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as transforms
from torchvision import datasets, models,transforms
from torchsummary import summary
from tqdm import tqdm
from utils.metrics import AverageMeter
from abc import abstractmethod
from os import path
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import itertools
import argparse
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

RESULTS_BASE_PATH = "output/results/"
MODELS_DIR = "output/models/"
DATA_SPLIT = [0.7, 0, 0.3]  # Order: Train, Validation, Test. Values between 0 and 1.
SOURCE = "cbis_ddsm"  

def write_line_to_csv(dir_path, file, data_row):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + file
    file_exists = Path(file_path).is_file()

    if file_exists:
        file_csv = open(file_path, 'a')
    else:
        file_csv = open(file_path, 'w')

    writer = csv.DictWriter(file_csv, delimiter=',', fieldnames=[*data_row], quoting=csv.QUOTE_NONE)

    if not file_exists:
        writer.writeheader()

    writer.writerow(data_row)

    file_csv.flush()
    file_csv.close()
    
def pre_datasets(run, np_images,label, classes): 
    images = np_images.copy()
    labels_save = label
    
    #The real classes must also be organized in the specific order
    classes_class = []
    for k in range(0,len(labels_save)):
        if(labels_save[k] in (classes[(classes['abnormality']=='calcification') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_class.append(0)
        elif(labels_save[k] in (classes[ (classes['abnormality']=='calcification') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_class.append(1)
        elif(labels_save[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_class.append(0)
        elif(labels_save[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_class.append(1)

    classes_density = []
    for k in range(0,len(labels_save)):
        if(labels_save[k] in (classes[ (classes['density']==0)])['name'].values):
            classes_density.append(0)
        elif(labels_save[k] in (classes[ (classes['density']==1)])['name'].values):
            classes_density.append(1)
        elif(labels_save[k] in (classes[ (classes['density']==2)])['name'].values):
            classes_density.append(2)
        elif(labels_save[k] in (classes[ (classes['density']==3)])['name'].values):
            classes_density.append(3)
        elif(labels_save[k] in (classes[ (classes['density']==4)])['name'].values):
            classes_density.append(4)
    
    classes_class = np.array(classes_class); classes_density = np.array(classes_density)
    
    return(images, classes_class, classes_density)  
    
from torchvision.models import resnet50, ResNet50_Weights

class CBIS_DDSM_classifier(nn.Module):
    
    def __init__(self, name: str = "model"):
        super(CBIS_DDSM_classifier, self).__init__()

        weights = ResNet50_Weights.DEFAULT
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(180), transforms.RandomVerticalFlip()])

        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.name = name
        self.checkpoints_files = []

        self.resnet50.fc = nn.Linear(512 * 4, 2)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
        """
        One epoch of the training loop
        Args:
            device: device where tensor manipulations are done
            dataloader: training set dataloader
            optimizer: training optimizer

        Returns:
            average loss on the training set
        """
        self.train()
        train_loss = []
        loss_meter = AverageMeter("Loss")
        train_bar = tqdm(dataloader, unit="batch", leave=False)
        for image_batch, label_batch in train_bar:
            image_batch = torch.stack([image_batch,image_batch,image_batch],1)
            image_batch = self.transform(image_batch) 
            image_batch = self.transforms(image_batch) 
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(image_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), len(image_batch))
            train_bar.set_description(f"Training Loss {loss_meter.avg:.3g}")
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test(
        self, device: torch.device, dataloader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        Testing the model on a separate testing set
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader

        Returns:
            average loss and accuracy on the training set
        """
        self.eval()
        test_loss = []
        test_acc = []
        labels = []
        with torch.no_grad():
            for image_batch, label_batch in dataloader:
                image_batch = torch.stack([image_batch,image_batch,image_batch],1)
                image_batch = self.transform(image_batch) 
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(image_batch)
                loss = self.criterion(pred_batch, label_batch)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1))
                    .cpu()
                    .numpy()
                    / len(label_batch)
                )
                labels.append(torch.argmax(pred_batch, dim=-1).cpu().numpy())
        flat_list = [item for sublist in labels for item in sublist]

        return np.mean(test_loss), np.mean(test_acc), flat_list

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        lr: int = 1e-04, 
        n_epoch: int = 100,
        patience: int = 20,
        checkpoint_interval: int = -1,
    ) -> None:
        """
        Fit the classifier on the training set
        Args:
            device: device where tensor manipulations are done
            train_loader: training set dataloader
            test_loader: test set dataloader
            save_dir: path where checkpoints and model should be saved
            lr: learning rate
            n_epoch: maximum number of epochs
            patience: optimizer patience
            checkpoint_interval: number of epochs between each save

        Returns:

        """

        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-05)

        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
            )

            print(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
            )
        self.cpu()
        self.save(save_dir)
        self.to(device)
        print('Saving the model')

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        self.save_metadata(directory)
        path_to_model = directory + "/" + (self.name + ".pt")
        torch.save(self.state_dict(), path_to_model)
        
    
def train_model(run,train_loader, test_loader): #val_loader
  if not os.path.exists("models"):
    os.makedirs("models")
  model = CBIS_DDSM_classifier("model"+ str(run))
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.fit(device,train_loader, "models")

  #Get the best model from the previous run
  model = CBIS_DDSM_classifier("model"+ str(run))
  model.load_state_dict(torch.load("models/model"+ str(run) +".pt"), strict=False)
  model.to(device)
  model.eval()

  test_loss, test_acc, test_labels = model.test(device, test_loader)

  print(
      f"Test Loss {test_loss:.3g} \t"
      f"Test Accuracy {test_acc * 100:.3g}% \t "
  )
  return(test_labels)
                
print("Images are being processed...")
images = np.load('CBIS_DDSM.npy')
label = np.load('CBIS_DDSM_labels.npy')
classes = pd.read_csv('CBIS_DDSM_description_all_concepts.csv')
classes2 = pd.read_csv('CBIS_DDSM_description_clean.csv')


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
             
run = 0
images, classes_class, classes_density= pre_datasets(images,label, classes2)

for train_index, test_index in skf.split(images, classes_density):
    x_train, x_test = images[train_index], images[test_index]
    y_train, y_test = classes_class[train_index], classes_class[test_index]

    x_train, y_train = del0(x_train,y_train, classes_density[train_index])

    tensor_x = torch.Tensor(np.squeeze(x_train))
    tensor_y = torch.Tensor(y_train).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x,tensor_y) 
    train_loader = DataLoader(my_dataset, batch_size=32,shuffle=True)

    tensor_x = torch.Tensor(np.squeeze(x_test))
    tensor_y = torch.Tensor(y_test).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x,tensor_y) 
    test_loader = DataLoader(my_dataset, batch_size=1)

    test_labels = train_model(run,train_loader, test_loader)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=test_labels)

    print('Accuracy: %.3f' % accuracy_score(y_test, test_labels))
    print('Precision: %.3f' % precision_score(y_test, test_labels,average='macro'))
    print('Recall: %.3f' % recall_score(y_test, test_labels,average='macro')) 

    run = run +1

    write_line_to_csv(
      "results/","Test_cbis_ddsm_10k.csv",
            {
                "RUN": (run),
                "Accuracy": accuracy_score(y_test, test_labels),
                "Precision": precision_score(y_test, test_labels,average='macro'),
                "Recall": recall_score(y_test, test_labels,average='macro')
             })           
