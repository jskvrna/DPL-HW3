import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from model import MyModel

IMAGE_SIZE = 224  # fixed by the assignment: the grader feeds 224x224 images
# Per-channel mean and std of the train set (R, G, B) for images in [0, 1]. They are given: use them in
# Normalize(MEAN, STD) in the transforms below.
MEAN = [0.4596, 0.4234, 0.3712]
STD = [0.2474, 0.2379, 0.2368]


def get_device():
    """Get the device to be used for training (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FungiDataset(Dataset):
    """Fungi dataset: `<root>/<split>/<class_folder>/<image>.jpg`.

    Args:
        root (str): Dataset directory containing the `train/`, `val/` and `test/` folders.
        split (str): One of "train", "val", "test".
        train (bool): True for the training set (training transform with augmentation), False for the
            evaluation transforms (val/test).
    """

    def __init__(self, root, split, train):
        self.train = train
        ### TODO 1.1 ###
        # Sorted list of the class folder names in <root>/<split>. The label of a class is its index in this list.
        self.classes = ...
        ### TODO 1.2 ###
        # List of (image file path, label) tuples, one for every image in <root>/<split>/<class folder>/.
        self.samples = ...

    def __len__(self):
        """Return the total number of samples in the dataset."""
        ### TODO 1.3 ###
        # Return the number of samples in the dataset.
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Returns:
            tuple: (image, label)
            image (torch.Tensor): The transformed image tensor, shape [3, 224, 224].
            label (int): Index of the class.
        """
        ### TODO 1.4 ###
        # Load the image of the sample `idx` as RGB (some files are not RGB), preprocess it with
        # `get_train_transform()` if `self.train` else `get_eval_transform()` (both are defined below)
        # and return (image, label).
        raise NotImplementedError


def get_train_transform():
    """Transform for training images: PIL image -> normalized float tensor [3, 224, 224]."""
    ### TODO 2.1 ###
    # Build the training transform: resize + center crop to IMAGE_SIZE, augmentations of your choice, ToTensor and
    # Normalize(MEAN, STD). Mind the order: augmentations work on the PIL image, Normalize comes last.
    # Augmentation is a regularization: the model sees a different version of every image in every epoch. Ideas:
    # random horizontal flip, random (resized) crop, color jitter, small rotations.
    raise NotImplementedError


def get_eval_transform():
    """Deterministic transform for val/test images (NO augmentation).

    The grader imports and calls this function (no arguments) to preprocess its images,
    so it must work standalone and return a callable: PIL image -> normalized float tensor [3, 224, 224].
    """
    ### TODO 2.2 ###
    # Build the evaluation transform: resize + center crop to IMAGE_SIZE, ToTensor and Normalize(MEAN, STD).
    raise NotImplementedError


def create_dataloaders(train_set, val_set, config):
    """Create the data loaders. Returns: (train_loader, val_loader)"""
    ### TODO 4.1 ###
    # Create the data loaders for `train_set` and `val_set` using `config["batch_size"]` and `config["num_workers"]`.
    # Think about: shuffling (which of the two loaders needs it?), pin_memory.
    train_loader = ...
    val_loader = ...
    return train_loader, val_loader


def setup_experiment_dir(config):
    """Create a timestamped experiment directory and set up logging.

    Returns: (exp_dir, logger)
    """
    exp_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    for sub in ("logs", "models"):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    with open(os.path.join(exp_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(exp_dir, "logs", "experiment.log")),
                  logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir, logger


def confusion_matrix(preds, labels, num_classes):
    """Confusion matrix [num_classes, num_classes]: entry [i, j] = number of images of the true class i predicted as j."""
    return torch.bincount(labels * num_classes + preds, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def macro_f1(confusion):
    """Macro F1: mean over the classes of the F1 score 2TP / (2TP + FP + FN), computed from the confusion matrix.

    A class that never occurs and is never predicted is skipped; one that is predicted but never occurs counts as 0.
    """
    tp = confusion.diag().float()
    f1 = 2 * tp / (confusion.sum(dim=0) + confusion.sum(dim=1))  # (TP + FP) + (TP + FN) = column sum + row sum
    return f1.nanmean().item()


def mean_average_precision(scores, labels, num_classes):
    """mAP: mean over the classes of the one-vs-rest average precision of the class scores.

    For every class the images are ranked by the score of that class; the average precision is the mean of the
    precision at the rank of every image of that class. scores: [N, num_classes], labels: LongTensor [N].
    """
    order = scores.argsort(dim=0, descending=True)  # [N, C]: image ranking for every class
    hits = torch.nn.functional.one_hot(labels, num_classes).gather(0, order).float()  # 1 where the ranked image is of the class
    precision = hits.cumsum(dim=0) / torch.arange(1, len(labels) + 1, device=scores.device)[:, None]
    return ((precision * hits).sum(dim=0) / hits.sum(dim=0)).nanmean().item()  # classes without images are skipped


### TODO 6.1 ###
# Evaluation needs no gradients: make this function run without gradient tracking (hint: a decorator from torch).
def evaluate(model, loader, criterion, device, num_classes):
    """Evaluate the model on a loader.

    Returns: dict with "loss", "acc", "macro_f1", "map" (mean average precision) and "confusion" (confusion matrix
    [num_classes, num_classes], rows = true class)
    """
    ### TODO 6.2 ###
    # Switch the model to evaluation mode (it changes the behaviour of BatchNorm, Dropout, ...).
    total_loss = 0.0
    all_preds, all_scores, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="Validation", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        ### TODO 6.3 ###
        # Make predictions for the batch and compute the loss.
        outputs = ...
        loss = ...
        total_loss += loss.item() * labels.size(0)
        all_preds.append(outputs.argmax(dim=1))
        all_scores.append(outputs.softmax(dim=1))
        all_labels.append(labels)

    preds, labels = torch.cat(all_preds), torch.cat(all_labels)
    accuracy = (preds == labels).float().mean().item()
    confusion = confusion_matrix(preds, labels, num_classes).cpu()
    return {"loss": total_loss / len(labels), "acc": accuracy,
            "macro_f1": macro_f1(confusion), "map": mean_average_precision(torch.cat(all_scores), labels, num_classes),
            "confusion": confusion}


def run_training(train_loader, val_loader, config, exp_dir, logger, model, wandb_run, device):
    """Train the model and validate it every `validate_every` epochs.

    Returns:
        float: Best validation accuracy achieved during training.
    """
    best_val_acc = 0.0
    ### TODO 5.1 ###
    # Define the loss function, the optimizer and (optionally) a learning rate scheduler. The scheduler is stepped
    # once per epoch below; if yours has to be stepped after every batch, move `scheduler.step()` into the batch loop.
    # Regularization to think about: weight decay (e.g. AdamW), label smoothing in the loss.
    criterion = ...
    optimizer = ...
    scheduler = None

    for epoch in range(config["num_epochs"]):
        ### TODO 5.2 ###
        # Switch the model to training mode.
        trn_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Train]",
                    leave=False, unit="batch")
        for images, labels in pbar:
            ### TODO 5.3 ###
            # Move the images and the labels to the device.

            ### TODO 5.4 ###
            # Reset the gradients of the optimizer.

            ### TODO 5.5 ###
            # Make predictions for the batch and compute the loss.
            # Optional speedup on modern GPUs (A16, A100, ...): run the forward pass and the loss inside
            # `with torch.autocast(device_type=device.type, dtype=torch.bfloat16):` (bfloat16 needs no GradScaler).
            outputs = ...
            loss = ...

            ### TODO 5.6 ###
            # Backpropagate the loss and make an optimizer step.

            trn_loss += loss.item() * labels.size(0)
            ### TODO 5.7 ###
            # Number of correctly classified images in this batch (an int).
            correct += ...
            total += labels.size(0)
            pbar.set_postfix({"Loss": f"{trn_loss / total:.4f}", "Acc": f"{correct / total:.4f}"})

        if scheduler is not None:
            scheduler.step()

        trn_loss /= total
        trn_acc = correct / total
        if wandb_run is not None:
            wandb_run.log({"train/loss": trn_loss, "train/acc": trn_acc, "lr": optimizer.param_groups[0]["lr"]},
                          step=epoch + 1)
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {trn_loss:.4f}, Train Accuracy: {trn_acc:.4f}")

        if (epoch + 1) % config["validate_every"] == 0:
            val = evaluate(model, val_loader, criterion, device, config["output_dim"])
            val_loss, val_acc = val["loss"], val["acc"]
            # the confusion matrix of every validation, plot it with `python visualize.py <file>.npy`
            os.makedirs(os.path.join(exp_dir, "confusion"), exist_ok=True)
            np.save(os.path.join(exp_dir, "confusion", f"epoch_{epoch + 1:03d}.npy"), val["confusion"].numpy())
            if wandb_run is not None:
                import matplotlib.pyplot as plt
                import wandb
                from visualize import confusion_figure
                figure = confusion_figure(val["confusion"], title=f"epoch {epoch + 1}")
                wandb_run.log({"val/loss": val_loss, "val/acc": val_acc,
                               "val/macro_f1": val["macro_f1"], "val/mAP": val["map"], "val/confusion": wandb.Image(figure)}, step=epoch + 1)
                plt.close(figure)
            logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
                        f"Macro F1: {val['macro_f1']:.4f}, mAP: {val['map']:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(getattr(model, "module", model).state_dict(), os.path.join(exp_dir, "models", "best_model.pth"))  # unwrap DataParallel
                logger.info(f"New best model saved (val accuracy {best_val_acc:.4f})")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="confs/config.yml")
    parser.add_argument("--overfit", type=int, metavar="N",
                        help="sanity check: train AND validate on the same N random training images (no augmentation), "
                             "the loss should go to ~0 and the accuracy to 100%%; N should be >= batch_size")
    parser.add_argument("--wandb", action="store_true", help="(optional) log the metrics to Weights & Biases")
    args = parser.parse_args()
    if sys.platform == "darwin":
        # macOS starts DataLoader workers with "spawn" by default, which re-imports torch in every worker at the start
        # of every epoch (and every validation): ~40 s each. "fork" starts them instantly.
        torch.multiprocessing.set_start_method("fork")

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["random_seed"])
    exp_dir, logger = setup_experiment_dir(config)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    device = get_device()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="dpl-hw3-fungi", name=os.path.basename(exp_dir), config=config, dir=exp_dir)

    train_set = FungiDataset(config["data_dir"], "train", train=True)
    val_set = FungiDataset(config["data_dir"], "val", train=False)
    logger.info(f"Train samples: {len(train_set)}, validation samples: {len(val_set)}")
    assert len(train_set.classes) == config["output_dim"]
    if args.overfit:
        eval_set = FungiDataset(config["data_dir"], "train", train=False)
        train_set = val_set = Subset(eval_set, torch.randperm(len(eval_set))[:args.overfit].tolist())
        logger.info(f"OVERFIT TEST: training and validating on the same {len(train_set)} training images")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # `batch_size` and `num_workers` in the config are per GPU, nn.DataParallel splits every batch between the GPUs
        config["batch_size"] *= num_gpus
        config["num_workers"] *= num_gpus
        logger.info(f"Using {num_gpus} GPUs (nn.DataParallel): batch size {config['batch_size']} "
                    f"({config['batch_size'] // num_gpus} per GPU), {config['num_workers']} data loading workers")
    train_loader, val_loader = create_dataloaders(train_set, val_set, config)

    model = MyModel(output_size=config["output_dim"]).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = run_training(train_loader, val_loader, config, exp_dir, logger, model, wandb_run, device)
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
