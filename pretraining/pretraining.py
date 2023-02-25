import configs
from data_utils import ImageSubset, ImageDataset
from model_utils import vgg16_model
import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models



def prepare_data (dataset_path, images_path=None, labels_path=None): 
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(configs.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.norm_mean, std=configs.norm_std)
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(configs.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.norm_mean, std=configs.norm_std)
    ])

    train_set = None
    valid_set = None
    train_dataset_dir = dataset_path
    valid_dataset_dir = dataset_path

    if configs.dataset_pure_name in ['places']:
        train_dataset_dir = os.path.join(train_dataset_dir, 'train')
        valid_dataset_dir = os.path.join(valid_dataset_dir, 'train')   # Because the validation set is very small, we just use and split the train set
    elif configs.dataset_pure_name in ['imagenette', 'imagewoof']:
        train_dataset_dir = os.path.join(train_dataset_dir, 'train')
        valid_dataset_dir = os.path.join(train_dataset_dir, 'val')
    elif configs.dataset_pure_name in ['cifar10']:
        train_dataset_dir = './data'
        valid_dataset_dir = './data'

    # Loading or computing the train and validation sets for different datasets: 
    if configs.dataset_pure_name in ['indoor', 'places', 'imagenet']:
        train_set = datasets.ImageFolder(root=train_dataset_dir, transform=train_transform)
        valid_set = datasets.ImageFolder(root=valid_dataset_dir, transform=valid_transform)

        data_size = len(train_set)
        indexes = list(range(data_size))
        np.random.shuffle(indexes)
        train_size = int(configs.train_ratio * data_size)
        train_indexes = indexes[:train_size]
        valid_indexes = indexes[train_size:]

        train_set = ImageSubset(train_set, train_indexes)
        valid_set = ImageSubset(valid_set, valid_indexes)

    elif configs.dataset_pure_name in ['imagenette', 'imagewoof']:
        train_set = datasets.ImageFolder(root=train_dataset_dir, transform=train_transform)
        valid_set = datasets.ImageFolder(root=valid_dataset_dir, transform=valid_transform)

    elif configs.dataset_pure_name == 'cifar10':
        train_set = datasets.CIFAR10(root=train_dataset_dir, train=True, transform=train_transform, download=True)
        valid_set = datasets.CIFAR10(root=valid_dataset_dir, train=False, transform=valid_transform, download=True)

    else: 
        data = pd.read_csv(labels_path)
        file_names = data['File'].tolist()
        labels = data['Label'].tolist()

        train_file_names, valid_file_names, train_labels, valid_labels = train_test_split(file_names, labels, test_size= 1.0 - configs.train_ratio, 
                                                                                          random_state=configs.random_seed, stratify=labels)
        
        train_set = ImageDataset(images_path, train_file_names, train_labels, transform=train_transform)
        valid_set = ImageDataset(images_path, valid_file_names, valid_labels, transform=valid_transform)

    # In case we want to focus on selected target classes among all the classes in the dataset: 
    if (configs.target_classes != None) and (len(configs.target_classes) > 0):
        class_indexes_dict = train_set.class_to_idx
        train_indexes = train_set.targets
        valid_indexes = valid_set.targets

        target_class_indexes = [v for k,v in class_indexes_dict.items() if k in configs.target_classes]
        target_train_indexes = [i for i,x in enumerate(train_indexes) if x in target_class_indexes]
        target_valid_indexes = [i for i,x in enumerate(valid_indexes) if x in target_class_indexes]
        print('target_class_indexes:', target_class_indexes)
        print('target_train_indexes: {}, target_valid_indexes: {}'.format(len(target_train_indexes), len(target_valid_indexes)))

        train_set = ImageSubset(train_set, target_train_indexes)
        valid_set = ImageSubset(valid_set, target_valid_indexes)

    train_loader = DataLoader(train_set, batch_size=configs.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=configs.train_batch_size, shuffle=True)
    
    return train_loader, valid_loader



def save_data_subset (data_subset, target_dataset_dir):
    images_info = data_subset.samples
    class_to_idx = data_subset.class_to_idx
    #print('class_to_idx:', class_to_idx)
    class_names = list(class_to_idx.keys())
    idx_to_class_dir = {}

    if not os.path.exists(target_dataset_dir):
        #print('Making target dataset dir:', target_dataset_dir)
        os.makedirs(target_dataset_dir)

    for cname in class_names:
        idx = class_to_idx[cname]
        target_class_dir = os.path.join(target_dataset_dir, cname)
        idx_to_class_dir[idx] = target_class_dir
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

    #print('idx_to_class_dir:', idx_to_class_dir)

    for i,(path, label) in enumerate(images_info):
        target_class_dir = idx_to_class_dir[label]
        _, fname = os.path.split(path)
        target_path = os.path.join(target_class_dir, fname)
        shutil.copy(path, target_path)



def pretrained_model (base_model_file=None):
    # When loading the model from disk, the "pretrained" parameter should be false and "num_classes" should be set based on the classes in the loaded model file
    # When not loading the model from disk, the "pretrained" parameter should be true, and "num_classes" should not be provided, 
    # or it should be equal to the classes in the pretrained model (e.g. ImageNet in PyTorch)
    pretrained = (not configs.load_model_from_disk)
    model = None
    partial_target_layers = []

    if configs.model_name == 'vgg16':
        model = vgg16_model(pretrained=pretrained, num_classes=configs.base_num_classes)
    else:
        model = models.__dict__[configs.model_name](pretrained=pretrained, num_classes=configs.base_num_classes)

    if configs.load_model_from_disk:
        checkpoint = torch.load(base_model_file)
        statedict = checkpoint
        if 'state_dict' in checkpoint:
            statedict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(statedict)

    if configs.model_name == 'resnet18':
        partial_target_layers = ['layer3', 'layer4', 'avgpool']

    if configs.pretrain_mode == 'partial_fine_tuning':
        for name, child in model.named_children():
            if name in partial_target_layers:
                for p in child.parameters():
                    p.requires_grad = True
            else:
                for p in child.parameters():
                    p.requires_grad = False
    elif configs.pretrain_mode == 'feature_extraction':
        for p in model.parameters():
            p.requires_grad = False

    if configs.model_name.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, configs.num_classes)
    elif configs.model_name == 'alexnet':
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, configs.num_classes)
    elif configs.model_name == 'vgg16':
        model.classifier.fc8a = nn.Linear(model.classifier.fc8a.in_features, configs.num_classes)
        
    return model



def train_epoch (model, optimizer, criterion, train_loader, valid_loader):
    model.train()
    train_loss = 0
    train_acc = 0
    train_steps = 0
    train_size = 0
    
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.cuda()
        labels = labels.cuda()
    
        optimizer.zero_grad()
        
        #labels = labels.unsqueeze(1).float()   # reshape to 2 dimensions
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        loss = loss.cpu()
        train_loss += loss.item()
        _, preds = torch.max(output, 1)
        preds = preds.cpu()
        labels = labels.cpu()
        train_acc += (preds == labels).float().sum()

        train_steps += 1
        train_size += len(preds)
    
    train_loss = train_loss / train_steps
    train_acc = train_acc / train_size   #len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    val_acc = 0
    val_steps = 0
    val_size = 0
    
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(valid_loader)):
            images = images.cuda()
            labels = labels.cuda()
            
            #labels = labels.unsqueeze(1).float()
            output = model(images)
            vloss = criterion(output, labels)
            
            vloss = vloss.cpu()
            val_loss += vloss.item()
            
            _, preds = torch.max(output, 1)
            preds = preds.cpu()
            labels = labels.cpu()
            val_acc += (preds == labels).float().sum()

            val_steps += 1
            val_size += len(preds)
            
    val_loss = val_loss / val_steps
    val_acc = val_acc / val_size   #len(valid_loader.dataset)
    
    return train_loss, train_acc, val_loss, val_acc



def train (model, model_path, train_loader, valid_loader): 
    model_params = [p for p in model.parameters() if p.requires_grad]
    #print('Number of params to learn:', len(model_params))
    
    optimizer = optim.Adam(model_params, lr=0.001, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0
    best_epoch = 0
    no_progress = 0
    
    for e in range(configs.epochs):
        print('\nEpoch {}/{}'.format(e+1, configs.epochs))
        train_loss, train_acc, val_loss, val_acc = train_epoch(model, optimizer, criterion, train_loader, valid_loader)
        
        print('loss: {:.3f} - acc: {:.3f} - val loss: {:.3f} - val acc: {:.3f}'.format(train_loss, train_acc, val_loss, val_acc))
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = e
            no_progress = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_progress += 1
            
        if no_progress >= configs.stop_patience:
            #print('Finished training in epoch {} because of no progress in {} consecutive epochs'.format(e, no_progress))
            break
    
    print('Best validation accuracy: {:.3f} (epoch {})'.format(best_acc, best_epoch))
    return train_losses, train_accs, val_losses, val_accs



def plot_results (train_losses, train_accs, val_losses, val_accs):
    # loss: 
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.show()
    
    # accuracy: 
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')
    plt.show()



def pretrain_model (full_dataset_path, dataset_path, base_model_file_path, model_file_path):
    print('----------------------------------------------')
    print('Pretraining the model ...')

    train_loader, valid_loader = prepare_data(full_dataset_path)
    save_data_subset(valid_loader.dataset, dataset_path)

    model = pretrained_model(base_model_file_path)
    model = model.cuda()

    train_losses, train_accs, val_losses, val_accs = train(model, model_file_path, train_loader, valid_loader)

    plot_results(train_losses, train_accs, val_losses, val_accs)

