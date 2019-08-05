import unittest
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from numpy import testing as nptest

from pytorchdl_gta5.dataset import GTA5
from pytorchdl_gta5.labels import GTA5Labels_TaskCV2017


class TestGTA5Labels(unittest.TestCase):
    def test_labelクラスからid_が取得出来る(self):
        labels = GTA5Labels_TaskCV2017()
        with self.subTest("Label: road"):
            self.assertEqual(0, labels.road.ID)
        with self.subTest("Label: sky"):
            self.assertEqual(10, labels.sky.ID)
        with self.subTest("Label: bicycle"):
            self.assertEqual(18, labels.bicycle.ID)

    def test_idの一覧(self):
        labels = GTA5Labels_TaskCV2017()
        expect = list(range(19))
        self.assertEqual(expect, labels.support_id_list)


class TestGTA5DataLoader(unittest.TestCase):
    def test_カラーマップをdecodeする(self):
        from dataclasses import dataclass
        @dataclass
        class Hoge:
            ID: int
            color: Tuple[int, int, int]

        label_map = [
            Hoge(1, (1, 1, 1)),
            Hoge(2, (2, 2, 2))
        ]

        lbl = np.array([
            [0, 1, 2],
            [0, 1, 2],
        ])
        expect = np.array([
            [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
            [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
        ])

        GTA5_ROOT = Path("./samples/data/")
        img_path = GTA5_ROOT / "labels/00001.png"
        color_lbl = GTA5._decode(lbl, label_map)

        nptest.assert_equal(color_lbl, expect)

    def test_getitemのisPathをTrueで画像のパスを取得する(self):
        ### preprocess
        GTA5_ROOT = Path("./samples/data/")
        dl = GTA5(root=GTA5_ROOT)

        for idx in range(3):
            img_path = GTA5_ROOT / "images" / f"{idx + 1:05}.png"
            with self.subTest(f"{idx}番目の画像パス"):
                self.assertEqual(img_path, dl.__getitem__(idx=idx, isPath=True)[0])
            lbl_path = GTA5_ROOT / "labels" / f"{idx + 1:05}.png"
            with self.subTest(f"{idx}番目のラベルパス"):
                self.assertEqual(lbl_path, dl.__getitem__(idx=idx, isPath=True)[1])

    def test_getitemで0番目の画像を取得する(self):
        ### preprocess
        GTA5_ROOT = Path("./samples/data/")
        img_path = GTA5_ROOT / "images/00001.png"
        expect = Image.open(str(img_path))
        expect = np.array(expect)

        dl = GTA5(root=GTA5_ROOT)
        img, _ = dl.__getitem__(idx=0)
        nptest.assert_equal(expect, img)

    def test_getitemで0番目のラベルを取得する(self):
        ### preprocess
        GTA5_ROOT = Path("./samples/data/")
        img_path = GTA5_ROOT / "labels/00001.png"
        expect = Image.open(str(img_path))
        expect = np.array(expect)

        dl = GTA5(root=GTA5_ROOT)
        _, lbl = dl.__getitem__(idx=0)
        nptest.assert_equal(expect, lbl)

if __name__ == '__main__':
    unittest.main()
