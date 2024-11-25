# TODO: Impelemnt this class similar to the one in jmesh/data/scannet.py
from jittor.dataset import Dataset
import jittor as jt
import numpy as np
import os
import glob
import json
from tqdm import tqdm
import trimesh
from pathlib import Path
import os.path as osp
from jmesh.utils.registry import DATASETS
from jmesh.data.tensor import MeshTensor
from jmesh.utils.general import summary_size, to_jt_var
from .utils import Compose
from jmesh.config import get_cfg

@DATASETS.register_module()
class CustomData(Dataset):
    classes = ['left_arm', 'right_arm', 'legs', 'torso', 'head']

    def __init__(self, dataroot,
                 transforms=None,
                 batch_size=1,
                 mode="train",
                 shuffle=False,
                 pattern = "scene*.obj",
                 num_workers=0,
                 level=6,
                 file_ext=".off",
                 color_aug = True,
                 feats = ["area","normal","center","angle","curvs","color"],
                 drop_last=False):

        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=drop_last)
        assert mode in ["train","val","test","trainval"]
        self.transforms = Compose(transforms)
        self.color_aug = color_aug

        self.files = self.browse_dataroot(dataroot,mode=mode,pattern=pattern)
        self.level = level

        self.feats = feats
        self.mode = mode
        self.total_len=len(self.files)
        self.use_max = True

    def browse_dataroot(self, dataroot, mode, pattern):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def collate_batch(self, batch):
        raise NotImplementedError

    def collate_levels(self, levels, centers):
        raise NotImplementedError

