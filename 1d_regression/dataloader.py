# %%
import torch
import os
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# %%
class TsDS(Dataset):
    def __init__(self, XL,yL,flatten=False,lno=None,long=True):
        self.samples=[]
        self.labels=[]
        self.flatten=flatten
        self.lno=lno
        self.long=long
        self.scaler = StandardScaler()
        for X,Y in zip(XL,yL):
            self.samples += [torch.tensor(X).float()]
            self.labels += [torch.tensor(Y)]
            
    def __len__(self):
        return sum([s.shape[0] for s in self.samples])

    def __getitem__(self, idx):
        if self.flatten: sample=self.samples[idx].flatten(start_dim=1)
        else: sample=self.samples[idx]
        if self.lno==None: label=self.labels[idx]
        elif self.long: label=self.labels[idx][:,self.lno].long()
        else: label=self.labels[idx][:,self.lno].float()
        return (sample,label)

    def fit(self,kind='seq'):
        if kind=='seq':
            self.lastelems=[torch.cat([s[:,-1,:] for s in self.samples],dim=0)]
            self.scaler.fit(torch.cat([le for le in self.lastelems],dim=0))            
        elif kind=='flat': self.scaler.fit(torch.cat([s for s in self.samples],dim=0))
    def scale(self,kind='flat',scaler=None):
        def cs(s):
            return (s.shape[0]*s.shape[1],s.shape[2])
        if scaler==None: scaler=self.scaler
        if kind=='seq':
            self.samples=[torch.tensor(scaler.transform(s.reshape(cs(s))).reshape(s.shape)).float() for s in self.samples]
            pass
        elif kind=='flat':
            self.samples=[torch.tensor(scaler.transform(s)).float() for s in self.samples]
    def unscale(self,kind='flat',scaler=None):
        def cs(s):
            return (s.shape[0]*s.shape[1],s.shape[2])
        if scaler==None: scaler=self.scaler
        if kind=='seq':
            self.samples=[torch.tensor(scaler.inverse_transform(s.reshape(cs(s))).reshape(s.shape)).float() for s in self.samples]
            pass
        elif kind=='flat':
            self.samples=[torch.tensor(scaler.inverse_transform(s)).float() for s in self.samples]

# %%
def get_numbers(name):
    splitted = name.split('_')
    g, d = (splitted[2]), int(splitted[3])
    return g, d

# %%
folder_path = os.path.join("marketdata")
l = os.listdir(folder_path)

data_type = "cs"
meta_train = {"train": [], "test": []}
meta_test = {"train": [], "test": []}

for file in l:
    if data_type in file:
        type_ = "train" if "train" in file else "test"
        g, d = get_numbers(file)
        if d < 20: # for meta-training
            meta_train[type_].append(file)
        else: # for meta-testing
            meta_test[type_].append(file)

# %%
meta_train["train"] = sorted(meta_train["train"])
meta_train["test"] = sorted(meta_train["test"])

data = list(zip(meta_train["train"], meta_train["test"]))

# %%
data = sorted(data, key=lambda x: get_numbers(x[0])[1])

# %%
# data

# %%
idx = 0

# %%
def load_task(task):
    """
    task is a tuple of strings of the form (train_cs_g_d_2.pkl, test_cs_g_d_2.pkl)
    returns X_train, y_train, X_test, y_test
    """
    train_file, test_file = task
    train_data = pickle.load(open(os.path.join(folder_path, train_file), "rb"))
    test_data = pickle.load(open(os.path.join(folder_path, test_file), "rb"))
    return train_data.samples, train_data.labels, test_data.samples, test_data.labels

# %%
def sample_task():
    global idx
    if idx >= len(data):
        idx = 0
    task = data[idx]
    idx += 1
    
    return load_task(task)

# %%
# for epoch in range(4):
#     for tasks in range(130):
#         X_train, y_train, X_test, y_test = sample_task()
        
#         for batch in zip(X_train, y_train):
#             X_tr, y_tr = torch.unsqueeze(batch[0],2), batch[1]
#             print(X_tr.shape, y_tr.shape)


