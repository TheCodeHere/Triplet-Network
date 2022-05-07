"""
Script that contains Siamese and TripleDataset classes that create combinations used to train those models
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from skimage import io
import numpy as np
from itertools import permutations, product


class KidneyStones(Dataset):
    """
    Class used to load all the entries of a csv file. The csv-file consists of two columns, 'image name' and 'label'.
    """
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), train: bool = True, img_view: str = "mixed", data_preload: bool =False, transform=None):
        self.root_dir = root
        self.transform = transform
        self.classes = ['Uric_acid','Brushite','Cystine','Struvite','Weddelite','Whewellite'] #Must be in order
        self.data = []
        self. sampleidx_by_label = {}

        csv_file = ""
        # create the path to point to an specific csv file
        if img_view.lower() == "section":
            csv_file = csv_file + "Section"
            self.root_dir = self.root_dir + r"/Section"
        elif img_view.lower() == "surface":
            csv_file = csv_file + "Surface"
            self.root_dir = self.root_dir + r"/Surface"
        else:
            csv_file = csv_file + "Mixed"
            self.root_dir = self.root_dir + r"/Mixed"

        if train:
            csv_file = csv_file + "-Train.csv"
            self.root_dir = self.root_dir + r"/train"
        else:
            csv_file = csv_file + "-Test.csv"
            self.root_dir = self.root_dir + r"/test"

        self.landmarks_frame = pd.read_csv(os.path.join(root, csv_file))

        self.targets = np.array(self.landmarks_frame['label'])
        self.samples_per_class = self.landmarks_frame.groupby('label').size().to_dict()
        self.idx_to_class = {i: _class for i, _class in zip(self.samples_per_class.keys(),self.classes)}

        # Returns a dictionary where the keys correspond to the label index, and each item is a tensor of the items
        # index for each label.
        self.split_by_label()

        if data_preload:
            self.data_preload()

    def data_preload(self):
        ld = DataLoader(self, batch_size=2400, shuffle=False, num_workers=0, pin_memory=True)

        for img, _ in ld:
            self.data += img

        self.data = torch.stack(self.data)

    def split_by_label(self):
        indexes = torch.tensor(range(len(self.targets)))

        for label in list(self.idx_to_class.keys()):
            self.sampleidx_by_label[label] = indexes[self.targets == label]

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = [r"/AU",r"/BRU",r"/CYS",r"/STR",r"/WD",r"/WW"]

        landmarks = self.landmarks_frame.iloc[idx, 1]
        landmarks = torch.tensor(landmarks)

        # img_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])
        img_path = os.path.join(self.root_dir + labels[landmarks], self.landmarks_frame.iloc[idx,0])
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, landmarks)


class SiameseDataset(Dataset):
    def __init__(self, dataset: KidneyStones, samples_per_perm: int = 2000, shuffle_prod: bool = True, transform=None):
        self.A = []
        self.B = []
        self.pair_labels = []
        self.dataset = dataset
        self.transform = transform
        self.sampleidx_by_label = dataset.sampleidx_by_label

        max_samp_per_class = min(dataset.samples_per_class.values()) // 2
        self.samples_per_perm = min(samples_per_perm, max_samp_per_class)

        lab_set_prod = list(product(dataset.idx_to_class.keys(), repeat=2))# [:3]
        if shuffle_prod:
            np.random.shuffle(lab_set_prod)
        #print("lab_set_prod = ", lab_set_prod)

        for i, j in lab_set_prod:
            a, b = self.Pairs_maker(self.sampleidx_by_label[i], self.sampleidx_by_label[j], i == j)
            self.A.extend(a)
            self.B.extend(b)
            self.pair_labels += [[i, j] for _ in range(self.samples_per_perm)]

        different_samples = len(set(self.A+self.B))

        print(f"Number of labels permutations: {len(lab_set_prod)}")
        print(f"Pair samples per permutations: {self.samples_per_perm}")
        print(f"Number of pair samples: {len(self.A)}")
        print(f"Number of different samples used: {different_samples}")

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        labels = [r"/AU",r"/BRU",r"/CYS",r"/STR",r"/WD",r"/WW"]
        
        class1, class2 = self.pair_labels[idx]

        landmarks = torch.tensor(self.pair_labels)[idx]

        img_path = os.path.join(self.dataset.root_dir + labels[class1],self.dataset.landmarks_frame.iloc[self.A[idx],0])
        A_sample = io.imread(img_path)

        img_path = os.path.join(self.dataset.root_dir + labels[class2],self.dataset.landmarks_frame.iloc[self.B[idx],0])
        B_sample = io.imread(img_path)

        if self.transform:
            A_sample = self.transform(A_sample)
            B_sample = self.transform(B_sample)

        return (A_sample, B_sample), landmarks

    '''
    def split_by_label(self):
        labels_set = list(self.dataset.idx_to_class.keys())

        sampleidx_by_label = {}
        batch_max_size = len(self.dataset)

        indexes = torch.tensor(range(len(self.dataset)))
        for label in labels_set:
            sampleidx_by_label[label] = indexes[self.dataset.targets == label]
            #print("sampleidx_by_label[",label,"] = ",sampleidx_by_label[label])
            batch_max_size = min(batch_max_size, len(sampleidx_by_label[label]))

        return sampleidx_by_label, labels_set, batch_max_size // 2
    '''

    def Pairs_maker(self, class_1, class_2, same_class):
        if same_class:
            index_a = np.random.choice(class_1, self.samples_per_perm * 2, replace=False)
            a = index_a[:self.samples_per_perm]
            b = index_a[self.samples_per_perm:]
        else:
            index_a = np.random.choice(class_1, self.samples_per_perm, replace=False)
            index_b = np.random.choice(class_2, self.samples_per_perm, replace=False)
            a = index_a
            b = index_b

        return a, b


class TripletDataset(Dataset):
    def __init__(self, dataset: KidneyStones, samples_per_perm: int = 2000, shuffle_perm: bool = True, transform=None):
        self.Anchor = []
        self.Positive = []
        self.Negative = []
        self.triplet_labels = []
        self.dataset = dataset
        self.transform = transform

        self.sampleidx_by_label = dataset.sampleidx_by_label

        max_samp_per_class = min(dataset.samples_per_class.values()) // 2
        self.samples_per_perm = min(samples_per_perm, max_samp_per_class)

        # Creates a list of tuples of two elements each
        lab_set_perm = list(permutations(dataset.idx_to_class.keys(), 2))#[:3]
        if shuffle_perm:
            np.random.shuffle(lab_set_perm)
        #print("lab_set_perm = ", lab_set_perm)

        # Now, given a label i and j, create triplets.
        for i, j in lab_set_perm:
            a, p, n = self.Triplets_maker(self.sampleidx_by_label[i], self.sampleidx_by_label[j])
            self.Anchor.extend(a)
            self.Positive.extend(p)
            self.Negative.extend(n)
            self.triplet_labels += [[i, j] for _ in range(self.samples_per_perm)]

        different_samples = len(set(self.Anchor + self.Positive + self.Negative))

        print(f"Number of labels permutations: {len(lab_set_perm)}")
        print(f"Triplet samples per permutations: {self.samples_per_perm}")
        print(f"Number of triplet samples: {len(self.Anchor)}")
        print(f"Number of different samples used: {different_samples}")

    def __len__(self):
        return len(self.Anchor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = [r"/AU", r"/BRU", r"/CYS", r"/STR", r"/WD", r"/WW"]
        class1, class2 = self.triplet_labels[idx]

        landmarks = torch.tensor(self.triplet_labels)[idx]

        img_path = os.path.join(self.dataset.root_dir + labels[class1],self.dataset.landmarks_frame.iloc[self.Anchor[idx], 0])
        anchor_sample = io.imread(img_path)

        img_path = os.path.join(self.dataset.root_dir + labels[class1],self.dataset.landmarks_frame.iloc[self.Positive[idx], 0])
        positive_sample = io.imread(img_path)

        img_path = os.path.join(self.dataset.root_dir + labels[class2],self.dataset.landmarks_frame.iloc[self.Negative[idx], 0])
        negative_sample = io.imread(img_path)

        if self.transform:
            anchor_sample = self.transform(anchor_sample)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)

        return (anchor_sample, positive_sample, negative_sample), landmarks

    def Triplets_maker(self, class_1, class_2):
        index_ap = np.random.choice(class_1, self.samples_per_perm * 2, replace=False)
        index_n = np.random.choice(class_2, self.samples_per_perm, replace=False)

        anchor = index_ap[:self.samples_per_perm]
        positive = index_ap[self.samples_per_perm:]
        negative = index_n

        return anchor, positive, negative

########################################################################################################################
class IndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# In order to use the Kidney stones library, you have to called it from your pytorch file as
