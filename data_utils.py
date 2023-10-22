import os
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets


class ImageDataset (Dataset):
    """ This class extends Dataset to allow reading and preprocessing an images dataset """

    def __init__(self, images_path, file_names, labels, transform):
        self.images_path = images_path
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        fname = self.file_names[index]
        path = os.path.join(self.images_path, fname)
        img = Image.open(path)
        
        if self.transform:
            img = self.transform(img)
        
        if self.labels is not None:
            label = self.labels[index]
            return img, label
        
        return img, fname


class ImageSubset (Subset):
    """ This class helps to be able to access class index, targets and file paths of the dataset, which are absent in the Subset class itself """

    def __init__(self, dataset, indexes):
        super().__init__(dataset, indexes)
        self.class_to_idx = dataset.class_to_idx
        self.targets = dataset.targets
        self.samples = [s for i,s in enumerate(dataset.samples) if i in indexes]


class CustomImageFolder (datasets.ImageFolder):
    """ This class extends ImageFolder to additionally return image paths """
    
    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
