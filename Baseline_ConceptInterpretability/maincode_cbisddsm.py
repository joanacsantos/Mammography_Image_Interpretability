import os
import time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,jaccard_score
import csv
import pathlib
from pathlib import Path
import random
import sys
import shutil
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as transforms
from torchvision import models, datasets, transforms
from sklearn.model_selection import StratifiedKFold
from torchsummary import summary
import logging
import json
from tqdm import tqdm
from utils.metrics import AverageMeter
from abc import abstractmethod
from os import path
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import itertools
import logging
import argparse
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from utils.plot import plot_concept_accuracy
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from sklearn.gaussian_process.kernels import Matern
from utils.robustness import Attacker
import seaborn as sns
import argparse
import textwrap
from utils.metrics import correlation_matrix

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
    order = range(0,images.shape[0])
    
    train, test = train_test_split(order, test_size=DATA_SPLIT[2], shuffle=True)
    train, val = train_test_split(train, test_size=0.143, shuffle=True)

    images_train = images[train]; images_test = images[test]; images_val = images[val]
    labels_save_test = [label[i] for i in test]; labels_save_train = [label[i] for i in train]; labels_save_val = [label[i] for i in val]

    #The real classes must also be organized in the specific order
    classes_test = []
    for k in range(0,len(labels_save_test)):
        if(labels_save_test[k] in (classes[(classes['abnormality']=='calcification') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_test.append(0)
        elif(labels_save_test[k] in (classes[ (classes['abnormality']=='calcification') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_test.append(1)
        elif(labels_save_test[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_test.append(0)
        elif(labels_save_test[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_test.append(1)
    classes_train = []
    for k in range(0,len(labels_save_train)):
        if(labels_save_train[k] in (classes[ (classes['abnormality']=='calcification') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_train.append(0)
        elif(labels_save_train[k] in (classes[ (classes['abnormality']=='calcification') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_train.append(1)
        elif(labels_save_train[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_train.append(0)
        elif(labels_save_train[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_train.append(1)
    classes_val = []
    for k in range(0,len(labels_save_val)):
        if(labels_save_val[k] in (classes[ (classes['abnormality']=='calcification') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_val.append(0)
        elif(labels_save_val[k] in (classes[ (classes['abnormality']=='calcification') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_val.append(1)
        elif(labels_save_val[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='BENIGN')])['name'].values):
            classes_val.append(0)
        elif(labels_save_val[k] in (classes[ (classes['abnormality']=='mass') & (classes['pathology']=='MALIGNANT')])['name'].values):
            classes_val.append(1)
    del images
    
    classes_train = np.array(classes_train); classes_test = np.array(classes_test); classes_val = np.array(classes_val)
    
    return(images_train, images_test, images_val, classes_train, classes_test, classes_val)  
    
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
        
    def forward(self, x):
        x = self.resnet50(x)
        return x

    def input_to_representation(self, x): 
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def representation_to_output(self, h):
        h = self.resnet50.fc(h)
        return h

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

    def test_epoch(
        self, device: torch.device, dataloader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        One epoch of the validating loop
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader

        Returns:
            average loss and accuracy on the training set
        """
        self.eval()
        test_loss = []
        test_acc = []
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

        return np.mean(test_loss), np.mean(test_acc)

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
        n_epoch: int = 200,
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
            test_loss, test_acc = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
                f"Val Loss {test_loss:.3g} \t"
                f"Val Accuracy {test_acc * 100:.3g}% \t "
            )

            print(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
                f"Val Loss {test_loss:.3g} \t"
                f"Val Accuracy {test_acc * 100:.3g}% \t "
            )
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
                print('Saving the model')
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info(f"Early stopping activated")
                break

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

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory + "/" + (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory + "/" + (self.name + ".json")
        metadata = {
            "name": self.name,
            "checkpoint_files": self.checkpoints_files,
        }
        with open(path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)

    def get_hooked_modules(self): #-> dict[str, nn.Module]:
        return {
            "Pool": self.resnet50.maxpool,
            "Layer1": self.resnet50.layer1,
            "Layer2": self.resnet50.layer2,
            "Layer3": self.resnet50.layer3,
            "Layer4": self.resnet50.layer4
        }
        
def concept_dataset(run,concept_name, concept_value, np_images,label, classes, train_slip, test_slip): 

    images = np_images.copy()
    order = np.array(range(0,images.shape[0]])
    random.shuffle(order)
    labels_save = [label[i] for i in order]
    images = images[order]
    
    #Divide the dataset in 2 parts: positives and negatives
    pos=[]; neg=[]
    labels_y = []
    for k in range(0,len(labels_save)):
        if(labels_save[k] in (classes[ (classes[concept_name]==1)])['name'].values):
            pos.append(k)
            labels_y.append(1)
        else:
            neg.append(k)
            labels_y.append(0)

    labels_y = np.array(labels_y)
    labels_save = np.array(labels_save)

    #The training and testing datasets have train_slip/2 and test_slip/2
            
    train_slip = int(train_slip/2)
    test_slip = int(test_slip/2)

    pos_train = pos[0:train_slip]
    pos_test = pos[train_slip:train_slip+test_slip]

    neg_train = neg[0:train_slip]
    neg_test = neg[train_slip:train_slip+test_slip]

    del pos, neg
    
    train = sorted(pos_train + neg_train)
    test = sorted(pos_test + neg_test)

    images_train = images[train]; images_test = images[test]
    labels_train = labels_y[train]; labels_test = labels_y[test]
    
    return(np.expand_dims(np.squeeze(images_train),axis=1), np.expand_dims(np.squeeze(images_test),axis=1), labels_train, labels_test)  
    
    
def concept_accuracy(
    images,label, classes, run,
    plot: bool,
    save_dir: Path = "results/cbis_ddsm/concept_accuracy",
    data_dir: Path = "data/cbis_ddsm",
    model_dir: Path = "models/",
    model_name: str = "model_",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    concept_names = ["density_1","density_2","density_3","density_4",
                     "birads_0","birads_2","birads_3","birads_4","birads_5",
                     "mass_shape_IRREGULAR","mass_shape_LOBULATED","mass_shape_OVAL",
                     "mass_margins_CIRCUMSCRIBED","mass_margins_ILL_DEFINED","mass_margins_SPICULATED",
                     "calc_type_PLEOMORPHIC","calc_distribution_CLUSTERED"]

    representation_dir = save_dir + "/" + f"{model_name}_representations"
    if not os.path.exists(representation_dir):
        os.makedirs(representation_dir)
    print("Starting Concept Accuracy")

    model_dir = model_dir #+ "/" +  model_name
    model = CBIS_DDSM_classifier(model_name)
    model.load_state_dict(torch.load(model_dir + "/" + f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name in concept_names:  
        if not os.path.exists(representation_dir):
          os.makedirs(representation_dir)
        logging.info(f"Working with concept {concept_name}")
        print(f"Working with concept {concept_name}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_train"
        )
        
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        X_train, X_test, y_train, y_test = concept_dataset(run,concept_name, concept_name, images,label, classes, 400, 60)

        X_train = np.stack([X_train,X_train,X_train],1)
        X_train = np.squeeze(X_train, 2)

        X_test = np.stack([X_test,X_test,X_test],1)
        X_test = np.squeeze(X_test, 2)

        model(transform(torch.from_numpy(X_train)).to(device))
        remove_all_hooks(handler_train_dic)
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(
            model, representation_dir, f"{concept_name}_test"
        )
        model(transform(torch.from_numpy(X_test)).to(device))
        remove_all_hooks(handler_test_dic)
        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Fitting concept classifiers for {module_name}")
            print(f"Fitting concept classifiers for {module_name}")

            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            car.fit(H_train, y_train)
            cav.fit(H_train, y_train)
            hook_name = f"{concept_name}_test_{module_name}"
            H_test = get_saved_representations(hook_name, representation_dir)
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAR",
                    accuracy_score(y_train, car.predict(H_train)),
                    accuracy_score(y_test, car.predict(H_test)),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAV",
                    accuracy_score(y_train, cav.predict(H_train)),
                    accuracy_score(y_test, cav.predict(H_test)),
                ]
            )
        results_df = pd.DataFrame(
            results_data,
            columns=["Concept", "Layer", "Method", "Train ACC", "Test ACC"],
            )
        csv_path = save_dir + "/" +  "metrics.csv"
        results_df.to_csv(csv_path, header=True, mode="w", index=False)
        shutil.rmtree(representation_dir)
    if plot:
        plot_concept_accuracy(save_dir, None, "cbis_ddsm" + str(run))
        
  
def wrap_labels(ax, width, break_long_words=False, do_y: bool = False) -> None:
    """
    Break labels in several lines in a figure
    Args:
        ax: figure axes
        width: maximal number of characters per line
        break_long_words: if True, allow breaks in the middle of a word
        do_y: if True, apply the function to the y axis as well

    Returns:

    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    if do_y:
        labels = []
        for label in ax.get_yticklabels():
            text = label.get_text()
            labels.append(textwrap.fill(text, width=width,
                                        break_long_words=break_long_words))
        ax.set_yticklabels(labels, rotation=0)

def plot_new_global_explanation(results_dir: Path, dataset_name: str, concept_categories: dict = None) -> None:
    sns.set(font_scale=1.2)
    sns.color_palette("colorblind")
    sns.set_style("white")
    metrics_df = pd.read_csv(results_dir +"/"+ "metrics.csv")
    concepts = list(metrics_df.columns[2:])
    classes = metrics_df["Class"].unique()
    methods = metrics_df["Method"].unique()
    plot_data = []
    for class_idx, concept, method in itertools.product(classes, concepts, methods):
        attr = np.array(metrics_df.loc[(metrics_df.Class == class_idx) & (metrics_df.Method == method)][concept])
        score = np.sum(attr)/len(attr)
        plot_data.append([method, class_idx, concept, score])
    plot_df = pd.DataFrame(plot_data, columns=["Method", "Class", "Concept", "Score"])
    tcar_scores = plot_df.loc[plot_df.Method == "TCAR"]["Score"]
    tcav_scores = plot_df.loc[plot_df.Method == "TCAV"]["Score"]
    #true_scores = plot_df.loc[plot_df.Method == "True Prop."]["Score"]
    #logging.info(f"TCAR-True Prop. Correlation: {np.corrcoef(tcar_scores, true_scores)[0, 1]:.2g}")
    if "TCAR Sensitivity" in methods:
        tcar_sensitivity_scores = plot_df.loc[plot_df.Method == "TCAR Sensitivity"]["Score"]
        logging.info(f"TCAR_Sensitivity-True Prop. Correlation: {np.corrcoef(tcar_sensitivity_scores, true_scores)[0, 1]:.2g}")
    #logging.info(f"TCAV-True Prop. Correlation: {np.corrcoef(tcav_scores, true_scores)[0, 1]:.2g}")
    for class_idx in classes:
      ax = sns.barplot(data=plot_df.loc[plot_df.Class == class_idx], x="Concept", y="Score", hue="Method")
      wrap_labels(ax, 10)
      plt.title(f"Class: {class_idx}")
      plt.ylim(bottom=0, top=1.1)
      plt.tight_layout()
      plt.savefig(results_dir + "/" + f"{dataset_name}_global_class{class_idx}.pdf")
      plt.close()

def global_explanations(
    images,label, classes, classes2, run,
    batch_size: int,
    plot: bool,
    save_dir: Path = "results/cbis_ddsm/global_explanations",
    data_dir: Path = "data/cbis_ddsm",
    model_dir: Path = "models/",
    model_name: str = "model1",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    concept_names = [ "density_1","density_2","density_3","density_4",
                     "birads_0","birads_2","birads_3","birads_4","birads_5",
                     "mass_shape_IRREGULAR","mass_shape_LOBULATED","mass_shape_OVAL",
                     "mass_margins_CIRCUMSCRIBED","mass_margins_ILL_DEFINED","mass_margins_SPICULATED",
                     "calc_type_PLEOMORPHIC","calc_distribution_CLUSTERED"]
    transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Starting Global Explanations")
    model_dir = model_dir 
    model = CBIS_DDSM_classifier(model_name)
    model.load_state_dict(torch.load(model_dir + "/" + f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()
    
    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_names]
    cav_classifiers = [CAV(device) for _ in concept_names]

    for concept_id in range(0,len(concept_names)):
        logging.info(f"Now fitting a CAR classifier for {concept_id}")
        print(f"Now fitting a CAR classifier for {concept_names[concept_id]}")
        images_train, images_test, labels_train, labels_test = concept_dataset(run,concept_names[concept_id],concept_names[concept_id], images,label, classes, 400, 60)

        images_train = np.stack([images_train,images_train,images_train],1)
        images_train = np.squeeze(images_train, 2)

        X_train = transform(torch.from_numpy(images_train)).to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = car_classifiers[concept_id]
        car.fit(H_train, labels_train)

        cav = cav_classifiers[concept_id]
        cav.fit(H_train, labels_train)

    a, images_test, b, c, classes_test, d = pre_datasets_aug(run,images,label, classes2)
    
    tensor_x_test = torch.Tensor(np.squeeze(images_test)) 
    tensor_y_test = torch.Tensor(classes_test).type(torch.LongTensor)

    my_dataset_test = TensorDataset(tensor_x_test,tensor_y_test) 
    test_loader = DataLoader(my_dataset_test, batch_size)

    logging.info("Producing global explanations for the test set")
    results_data = []
    for X_test, Y_test in tqdm(test_loader, unit="batch", leave=False):
        X_test = transform(X_test).to(device)
        X_test = torch.stack([X_test,X_test,X_test],1)
        H_test = model.input_to_representation(X_test).detach().cpu().numpy()
        pred_concepts = [car.predict(H_test) for car in car_classifiers]
        cav_preds = [
            cav.concept_importance(H_test, Y_test, 2, model.representation_to_output) ##latent_reps, labels, num_classes, rep_to_output
            for cav in cav_classifiers
        ]

        targets = [ [int(label in concept_names) for label in Y_test] for concept in concept_names]

        results_data += [
            ["TCAR", label.item()]
            + [pred_concept[example_id] for pred_concept in pred_concepts]
            for example_id, label in enumerate(Y_test)
        ]

        results_data += [
            ["TCAV", label.item()] + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(Y_test)
        ]

    results_df = pd.DataFrame(
        results_data, columns=["Method", "Class"] + [f"{i}" for i in concept_names]
    )
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir + "/" + "metrics.csv", index=False)

    if plot:
        plot_new_global_explanation(save_dir, "cbis_ddsm", None)
                
print("Images are being processed...")
images = np.load('CBIS_DDSM.npy')
label = np.load('CBIS_DDSM_labels.npy')
classes = pd.read_csv('CBIS_DDSM_description_all_concepts.csv')
classes2 = pd.read_csv('CBIS_DDSM_description_clean.csv')

for run in range(0,30):  # Go until 30
  print("Run: " + str(run + 1) + "...")

  images_train, images_test, images_val, classes_train, classes_test, classes_val = pre_datasets(run,images,label, classes2)
    
  tensor_x = torch.Tensor(np.squeeze(images_train))
  tensor_y = torch.Tensor(classes_train).type(torch.LongTensor)

  my_dataset = TensorDataset(tensor_x,tensor_y) 
  train_loader = DataLoader(my_dataset, batch_size=128,shuffle=True)

  tensor_x_val = torch.Tensor(np.squeeze(images_val)) 
  tensor_y_val = torch.Tensor(classes_val).type(torch.LongTensor)

  my_dataset_val = TensorDataset(tensor_x_val,tensor_y_val) 
  val_loader = DataLoader(my_dataset_val, batch_size=128,shuffle=True)

  if not os.path.exists("models/cbisddsm"):
      os.makedirs("models/cbisddsm")
  model = CBIS_DDSM_classifier("model"+ str(run))
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.fit(device,train_loader, val_loader, "models/cbisddsm")

  tensor_x = torch.Tensor(np.squeeze(images_test))
  tensor_y = torch.Tensor(classes_test).type(torch.LongTensor)

  my_dataset = TensorDataset(tensor_x,tensor_y) 
  test_loader = DataLoader(my_dataset, batch_size=1)

  #Get the best model from the previous run
  model = CBIS_DDSM_classifier("model"+ str(run))
  model.load_state_dict(torch.load("models/cbisddsm/model"+ str(run) +".pt"), strict=False)
  model.to(device)
  model.eval()

  test_loss, test_acc, test_labels = model.test(device, test_loader)

  print('Accuracy: %.3f' % accuracy_score(classes_test, test_labels))
  print('Precision: %.3f' % precision_score(classes_test, test_labels))
  print('Recall: %.3f' % recall_score(classes_test, test_labels)) 

  write_line_to_csv(
      "results/","Test_CBIS_DDSM.csv",
            {
                "RUN": (run + 1),
                "Accuracy": accuracy_score(classes_test, test_labels),
                "Precision": precision_score(classes_test, test_labels),
                "Recall": recall_score(classes_test, test_labels)
             })  
  concept_accuracy(images,label,classes,run,True,"results/cbis_ddsm" + str(run) + "/concept_accuracy","data/cbis_ddsm","models/cbisddsm/","model" + str(run))

  global_explanations(images,label,classes, classes2,run,1,True,"results/cbis_ddsm" + str(run) + "/global_explanation","data/cbis_ddsm","models/cbisddsm/","model" + str(run))
    