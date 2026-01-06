import os
import sys
import time
import random
import argparse
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from models.spo_clapscore import SPO_CLAPScore
from datasets.xacle_dataset_standard import get_spodataset
from losses.loss_function import get_loss_function
import utils.utils as utils


def seed_everything(seed: int = 0):
    """
    Fix seeds for PyTorch, NumPy, and Python random to ensure reproducibility.
    """
    # Fix PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Fix NumPy seed
    np.random.seed(seed)

    # Fix Python random seed
    random.seed(seed)

    # Environment variables for stricter reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CuDNN settings (use deterministic algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_trainable_param_summary(model):
    total, trainable = 0, 0
    print("---- Trainable parameters ----")
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"[TRAIN] {n}  shape={tuple(p.shape)}")
    print(f"Trainable: {trainable:,} / Total: {total:,}")


def train(cfg):
    # -------- Initial setup --------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    chkpt_dir = os.path.join(
        cfg["output_dir"], f"{now}_humanm2dclap_lr{cfg['lr']}_seed{cfg['seed']}"
    )
    os.makedirs(chkpt_dir, exist_ok=True)
    utils.json_dump(os.path.join(chkpt_dir, "config.json"), cfg)

    log_txt_path = os.path.join(chkpt_dir, "log.txt")
    sys.stdout = utils.Logger(log_txt_path)

    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"])

    # -------- Dataset / DataLoader --------

    train_ds = get_spodataset(
        cfg["train_list"],
        os.path.join(cfg["wav_dir"], "train"),
        max_sec=cfg["max_len"],
        sr=cfg["audio_encoder"]["sample_rate"],
    )
    val_ds = get_spodataset(
        cfg["validation_list"],
        os.path.join(cfg["wav_dir"], "validation"),
        max_sec=cfg["max_len"],
        sr=cfg["audio_encoder"]["sample_rate"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=train_ds.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=val_ds.collate_fn,
    )

    # -------- Model / Loss / Optimizer --------
    model = SPO_CLAPScore(cfg, device).to(device)
    loss_fn = get_loss_function(cfg["loss"])

    # ---- LR Settings ----
    peak_lr = cfg["lr"]
    peak_epoch = cfg["lr_peak_epoch"]  # Warmup length
    max_epochs = cfg["epochs"]

    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)

    # Scheduler (timm CosineLRScheduler)
    scheduler = CosineLRScheduler(
        opt,
        t_initial=max_epochs,
        lr_min=0.0,
        warmup_t=peak_epoch,
        warmup_lr_init=0.0,
        warmup_prefix=True,
        t_in_epochs=True,
    )

    print_trainable_param_summary(model)
    best_srcc, patience = -np.inf, 0

    print(f"Train loader length: {len(train_loader)}")

    # -------- Training Loop --------
    for epoch in range(cfg["epochs"]):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for batch in train_pbar:
            batch = utils.move_to_device(batch, device)
            opt.zero_grad()

            # Forward
            pred = model.forward(batch)

            # Loss / Backward
            loss = loss_fn(pred, batch["scores"], batch["num_class"])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Monitor predictions (Train)
        train_pred = (pred.detach().cpu().numpy() * 5 + 5).tolist()
        train_gt = (batch["scores"].detach().cpu().numpy() * 5 + 5).tolist()

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        preds, gts = [], []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]")
            for batch in val_pbar:
                batch = utils.move_to_device(batch, device)
                pred = model.forward(batch)
                loss = loss_fn(pred, batch["scores"], batch["num_class"])
                val_loss += loss.item()
                preds.extend((pred.cpu().numpy() * 5 + 5).tolist())
                gts.extend((batch["scores"].cpu().numpy() * 5 + 5).tolist())

        avg_val_loss = val_loss / len(val_loader)
        srcc = spearmanr(gts, preds).correlation
        mse = np.mean((np.array(gts) - np.array(preds)) ** 2)

        # Monitor predictions (Val - last batch)
        val_pred = (pred.detach().cpu().numpy() * 5 + 5).tolist()
        val_gt = (batch["scores"].detach().cpu().numpy() * 5 + 5).tolist()

        elapsed_time = time.time() - start_time

        # ---------- epoch summary ----------
        print(
            f"Epoch {epoch:04d} completed in {elapsed_time:.2f} seconds | Train Loss : {avg_train_loss:.4f}\tVal Loss : {avg_val_loss:.4f}\tVal SRCC / MSE: {srcc:.4f} , {mse:.4f}"
        )
        # -----------------------------------

        ########################################################################
        train_pred = [f"{v: 05.2f}" for v in train_pred]
        train_gt = [f"{v: 05.2f}" for v in train_gt]
        val_pred = [f"{v: 05.2f}" for v in val_pred]
        val_gt = [f"{v: 05.2f}" for v in val_gt]
        print(f"\ttrain pred   : {train_pred}")
        print(f"\ttrain_gt     : {train_gt}")
        print(f"\tval pred     : {val_pred}")
        print(f"\tval_gt       : {val_gt}")
        ########################################################################

        # -------- Scheduler Step --------
        scheduler.step(epoch + 1)

        # -------- Early Stopping / Checkpoint --------
        if srcc > best_srcc:
            best_srcc = srcc
            patience = 0
            torch.save(model.state_dict(), os.path.join(chkpt_dir, "best_model.pt"))
            print("✅ Best model updated")
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print("⛔️ Early stopping (patience exhausted)")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SPO-CLAPScore")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file (e.g., cfg/config.json)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    config = utils.load_config(args.config)
    train(config)
