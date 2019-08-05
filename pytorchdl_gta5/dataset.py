from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset

from pytorchdl_gta5.labels import GTA5Labels_TaskCV2017


class GTA5(torchDataset):
    label_map = GTA5Labels_TaskCV2017()

    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        LBL_DIR_NAME = "labels"
        SUFFIX = ".png"

        def __init__(self, root):
            self.root = root
            self.img_paths = self.create_imgpath_list()
            self.lbl_paths = self.create_lblpath_list()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            lbl_path = self.lbl_paths[idx]
            return img_path, lbl_path

        def create_imgpath_list(self):
            img_dir = self.root / self.IMG_DIR_NAME
            img_path = [path for path in img_dir.glob(f"*{self.SUFFIX}")]
            return img_path

        def create_lblpath_list(self):
            lbl_dir = self.root / self.LBL_DIR_NAME
            lbl_path = [path for path in lbl_dir.glob(f"*{self.SUFFIX}")]
            return lbl_path

    def __init__(self, root: Path):
        """

        :param root: (Path)
            this is the directory path for GTA5 data
            must be the following
            e.g.)
                ./data
                ├── images
                │   ├── 00001.png
                │   ├── ...
                │   └── 24966.png
                ├── images.txt
                ├── labels
                │   ├── 00001.png
                │   ├── ...
                │   └── 24966.png
                ├── test.txt
                └── train.txt
        """
        self.root = root
        self.paths = self.PathPair_ImgAndLabel(root=self.root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, isPath=False):
        img_path, lbl_path = self.paths[idx]
        if isPath:
            return img_path, lbl_path

        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)
        return img, lbl

    @staticmethod
    def read_img(path):
        img = Image.open(str(path))
        img = np.array(img)
        return img

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)

    @staticmethod
    def _decode(lbl, label_map):
        # remap_lbl = lbl[np.where(np.isin(lbl, cls.label_map.support_id_list), lbl, 0)]
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl
