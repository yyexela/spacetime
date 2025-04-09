from typing import List
import os
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from dataloaders.datasets import SequenceDataset, default_data_path

def load_dataset_raw(dataset, matrix):
    """
    Load original unprocessed dataset
    """
    if dataset in ["ODE_Lorenz", "PDE_KS"]:
        train_mat = loadmat(f'../../Datasets/{dataset}/{matrix}')
        train_mat = train_mat[list(train_mat.keys())[-1]]
        test_mat = loadmat(f'../../Datasets/{dataset}/{matrix}')
        test_mat = test_mat[list(test_mat.keys())[-1]]

        train_mat = np.swapaxes(train_mat, 0, 1)
        test_mat = np.swapaxes(test_mat, 0, 1)

        train_mat = torch.Tensor(train_mat.astype(np.float32))
        test_mat = torch.Tensor(test_mat.astype(np.float32))
    else:
        raise Exception(f"Timeseries dataset {dataset} not found")
    return train_mat, test_mat

class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data, loc=None):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean

class CTFSequenceDataset(SequenceDataset):
    @property
    def n_tokens_time(self):
        # Shape of the dates: depends on `timeenc` and `freq`
        return self.dataset_train.n_tokens_time  # data_stamp.shape[-1]

    @property
    def d_input(self):
        return self.dataset_train.d_input

    @property
    def d_output(self):
        return self.dataset_train.d_output

    @property
    def l_output(self):
        return self.dataset_train.pred_len

    def _get_data_filename(self, variant):
        return self.variants[variant]

    @staticmethod
    def collate_fn(batch, resolution, **kwargs):
        x, y, *z = zip(*batch)
        x = torch.stack(x, dim=0)[:, ::resolution]
        y = torch.stack(y, dim=0)
        z = [torch.stack(e, dim=0)[:, ::resolution] for e in z]
        return x, y, *z

    def setup(self):
        self.data_dir = default_data_path.parent.parent.parent.parent / 'data' / self._name_ / 'train'

        self.dataset_train = self._dataset_cls(
            root_path=self.data_dir,
            flag="train",
            size=self.size,
            data_path=self._get_data_filename(self.variant),
            scale=self.scale,
            inverse=self.inverse,
        )

        self.dataset_val = self._dataset_cls(
            root_path=self.data_dir,
            flag="val",
            size=self.size,
            data_path=self._get_data_filename(self.variant),
            scale=self.scale,
            inverse=self.inverse,
        )

        self.dataset_test = self._dataset_cls(
            root_path=self.data_dir,
            flag="test",
            size=self.size,
            data_path=self._get_data_filename(self.variant),
            scale=self.scale,
            inverse=self.inverse,
        )

        # alexey
        #print(f"train len {len(self.dataset_train)}")
        #try:
            #print(f"val len {len(self.dataset_val)}")
        #except Exception as e:
            #print(e)
        #try:
            #print(f"test len {len(self.dataset_test)}")
        #except Exception as e:
            #print(e)

        pass

class CTFDataset(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        data_path="",
        scale=True,
        inverse=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.flag = flag
        self.set_type = type_map[flag]

        self.scale = scale
        self.inverse = inverse
        self.forecast_horizon = self.pred_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        # if data_path == 'national_illness.csv':
        #     breakpoint()

    def _borders(self, df_raw):
        num_train = int(len(df_raw) * 0.8)
        num_vali = len(df_raw) - num_train
        border1s = [0, num_train - self.seq_len, len(df_raw) - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        # alexey
        #print("borders:")
        #print(border1s)
        #print(border2s)
        return border1s, border2s

    def __read_data__(self):
        self.scaler = StandardScaler()

        train_mat = loadmat(self.root_path / self.data_path)
        train_mat = train_mat[list(train_mat.keys())[-1]]
        test_mat = loadmat(self.root_path / self.data_path)
        test_mat = test_mat[list(test_mat.keys())[-1]]

        train_mat = np.swapaxes(train_mat, 0, 1)
        test_mat = np.swapaxes(test_mat, 0, 1)

        train_mat = torch.Tensor(train_mat.astype(np.float32))
        test_mat = torch.Tensor(test_mat.astype(np.float32))

        df_data = pd.DataFrame(train_mat)

        border1s, border2s = self._borders(df_data)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)  # Scaled down, should not be Y
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

        self.data_stamp = np.arange(0,df_data.shape[0]).reshape(-1,1)

        # alexey
        #print(f"set type {self.set_type} x {self.data_x.shape} y {self.data_y.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x = np.concatenate(
            [seq_x, np.zeros((self.pred_len, self.data_x.shape[-1]))], axis=0
        )

        if self.inverse:
            # seq_y = np.concatenate(
            #     [
            #         self.data_x[r_begin : r_begin + self.label_len],
            #         self.data_y[r_begin + self.label_len : r_end],
            #     ],
            #     0,
            # )
            # raise NotImplementedError   # OLD in S4 codebase
            seq_y = self.data_y[s_end:r_end]
        else:
            # seq_y = self.data_y[r_begin:r_end] # OLD in Informer codebase
            seq_y = self.data_y[s_end:r_end]

        # OLD in Informer codebase
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        mark = self.data_stamp[s_begin:s_end]
        mark = np.concatenate([mark, np.zeros((self.pred_len, mark.shape[-1]))], axis=0)

        mask = np.concatenate([np.zeros(self.seq_len), np.zeros(self.pred_len)], axis=0)
        mask = mask[:, None]

        # Add the mask to the timestamps: # 480, 5
        # mark = np.concatenate([mark, mask[:, np.newaxis]], axis=1)

        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        mark = mark.astype(np.float32)
        mask = mask.astype(np.int64)

        return torch.tensor(seq_x), torch.tensor(seq_y), torch.tensor(mark), torch.tensor(mask)

    def __len__(self):
        # alexey
        #print(f"len for set type {self.set_type} is {len(self.data_x) - self.seq_len - self.pred_len + 1}")
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, loc=None):
        return self.scaler.inverse_transform(data, loc)

    @property
    def d_input(self):
        if self.features == 'M':
            return 1
        return self.data_x.shape[-1]

    @property
    def d_output(self):
        if self.features in ["M", "S"]:
            return self.data_x.shape[-1]
        elif self.features == "MS":
            return 1
        else:
            raise NotImplementedError

    @property
    def n_tokens_time(self):
        if self.freq == 'h':
            return [13, 32, 7, 24]
        elif self.freq == 't':
            return [13, 32, 7, 24, 4]
        else:
            raise NotImplementedError

class _Dataset_CTF(CTFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def n_tokens_time(self):
        assert self.freq == "h"
        return [13, 32, 7, 24]

class PDE_KD(CTFSequenceDataset):
    _name_ = "PDE_KS"

    _dataset_cls = _Dataset_CTF

    init_defaults = {
        "size": None,
        "features": "S",
        "target": "OT",
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    variants = {
        1: "X1train.mat",
        2: "X2train.mat",
        3: "X3train.mat",
        4: "X4train.mat",
        5: "X5train.mat",
        6: "X6train.mat",
        7: "X7train.mat",
        8: "X8train.mat",
        9: "X9train.mat",
        10: "X10train.mat"
    }

class PDE_KS(CTFSequenceDataset):
    _name_ = "PDE_KS"

    _dataset_cls = _Dataset_CTF

    init_defaults = {
        "size": None,
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    variants = {
        1: "X1train.mat",
        2: "X2train.mat",
        3: "X3train.mat",
        4: "X4train.mat",
        5: "X5train.mat",
        6: "X6train.mat",
        7: "X7train.mat",
        8: "X8train.mat",
        9: "X9train.mat",
        10: "X10train.mat"
    }

class ODE_Lorenz(CTFSequenceDataset):
    _name_ = "ODE_Lorenz"

    _dataset_cls = _Dataset_CTF

    init_defaults = {
        "size": None,
        "variant": 0,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    variants = {
        1: "X1train.mat",
        2: "X2train.mat",
        3: "X3train.mat",
        4: "X4train.mat",
        5: "X5train.mat",
        6: "X6train.mat",
        7: "X7train.mat",
        8: "X8train.mat",
        9: "X9train.mat",
        10: "X10train.mat"
    }

def load_data(config_dataset, config_loader):
    if config_dataset['_name_'] == "ODE_Lorenz":
        dataset = ODE_Lorenz(**config_dataset)
    elif config_dataset['_name_'] == "PDE_KS":
        dataset = PDE_KS(**config_dataset)
    dataset.setup()
    
    train_loader = dataset.train_dataloader(**config_loader)
    # Eval loaders are dictionaries where key is resolution, value is dataloader
    # - Borrowed from S4 dataloaders. For now just set resolution to 1
    val_loader   = dataset.val_dataloader(**config_loader)[None]
    test_loader  = dataset.test_dataloader(**config_loader)[None]
    return train_loader, val_loader, test_loader  # , dataset


def visualize_data(dataloaders, splits=['train', 'val', 'test'],
                   save=False, args=None, title=None):
    None