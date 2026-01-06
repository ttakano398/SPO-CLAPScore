import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad as pad1d
import pandas as pd
import os
import torchaudio
import numpy as np


def get_spodataset(txt_file_path, wav_dir, max_sec, sr, num_bins=10):
    return XACLESPODataset(txt_file_path, wav_dir, max_sec, sr, num_bins)


def get_spoinf_dataset(txt_file_path, wav_dir, max_sec, sr, num_bins=10):
    return XACLESPOINFDataset(txt_file_path, wav_dir, max_sec, sr, num_bins)


class XACLESPODataset(Dataset):
    def __init__(
        self,
        txt_file_path: str,
        wav_dir: str,
        max_sec: int = 10,
        sr: int = 16_000,
        num_bins: int = 10,
        use_stats: bool = False,
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.wav_max_len = int(max_sec * sr)

        # global_statsはcsvを通して共通なので1行目で初期化
        self.use_stats = use_stats
        if self.use_stats:
            first_row = df.iloc[0]
            self.standard_stats = [first_row["global_mean"], first_row["global_std"]]
        else:
            self.standard_stats = None

        min_score = df["average_score"].min()
        max_score = df["average_score"].max()

        bins = np.linspace(min_score, max_score, num_bins + 1)
        bins[0] -= 0.001

        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(num_bins)]

        df["rating_category"] = pd.cut(
            df["average_score"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

        df["num_class"] = df.groupby("rating_category")["rating_category"].transform(
            "count"
        )
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, _ = torchaudio.load(wav_path)
        mos = float(r["average_score"])
        caption = r["text"]
        num_class = int(r["num_class"])

        standard_stats = self.standard_stats

        return dict(
            wav=wav,
            score=mos,
            caption=caption,
            num_class=num_class,
            wav_path=wav_path,
            standard_stats=standard_stats,
        )

    def collate_fn(self, batch):
        wav_max_len = self.wav_max_len
        wavs = [b["wav"] for b in batch]
        wav_fixed = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        mos_score = torch.tensor([b["score"] for b in batch], dtype=torch.float)
        num_class = torch.tensor([b["num_class"] for b in batch], dtype=torch.long)
        if self.use_stats:
            standard_stats = torch.tensor(
                [b["standard_stats"] for b in batch], dtype=torch.float
            )
        else:
            standard_stats = None

        captions = [b["caption"] for b in batch]

        return dict(
            wavs=wav_batch,
            scores=mos_score,
            caption_tokens=captions,
            num_class=num_class,
            wav_paths=[b["wav_path"] for b in batch],
            standard_stats=standard_stats,
        )


class XACLESPOINFDataset(Dataset):
    def __init__(
        self,
        txt_file_path: str,
        wav_dir: str,
        max_sec: int = 10,
        sr: int = 16_000,
        num_bins: int = 10,
        use_stats: bool = False,
    ):
        super().__init__()
        df = pd.read_csv(txt_file_path)
        self.wav_dir = wav_dir
        self.wav_max_len = int(max_sec * sr)

        # global_statsはcsvを通して共通なので1行目で初期化
        self.use_stats = use_stats
        if self.use_stats:
            first_row = df.iloc[0]
            self.standard_stats = [first_row["global_mean"], first_row["global_std"]]
        else:
            self.standard_stats = None

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, r["wav_file_name"].lstrip("/"))
        wav, _ = torchaudio.load(wav_path)
        caption = r["text"]

        return dict(
            wav=wav,
            caption=caption,
            wav_path=wav_path,
        )

    def collate_fn(self, batch):
        wav_max_len = self.wav_max_len
        wavs = [b["wav"] for b in batch]
        wav_fixed = []
        for wav in wavs:
            pad_len = wav_max_len - wav.shape[1]
            if pad_len > 0:
                padded = pad1d(wav, (0, pad_len), mode="constant", value=0.0)
            else:
                padded = wav[:, :wav_max_len]
            wav_fixed.append(padded)
        wav_batch = torch.stack(wav_fixed)

        captions = [b["caption"] for b in batch]

        return dict(
            wavs=wav_batch,
            caption_tokens=captions,
            wav_paths=[b["wav_path"] for b in batch],
        )
