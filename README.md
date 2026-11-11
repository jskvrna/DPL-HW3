# 🍄 DPL HW3 - Fungi Image Classification

This repository contains the code for the third homework assignment of the **DPL course at CTU**. The task is to train a convolutional neural network **from scratch** that classifies photos of fungi into 40 species (the **fungi40c** dataset).

> ⏰ **Deadline: 2 December 2026, 23:59** (the homework and the optional tournament).
>
> ⚠️ **The basic part is without any pretrained architectures or weights**: no pretrained models and no transfer learning, everything is trained from scratch on the provided dataset (pretrained models are allowed only in the optional [tournament](#-tournament-additional-part)).
>
> 📦 **Size limit: every submission zip (the homework, and separately the tournament) must be at most 500 MB.**

---

## 📁 Repository Structure

```
hw3-cnns/
├── confs/config.yml     # Hyperparameters and paths
├── model.py             # Neural network architecture definition
├── train.py             # Dataset, preprocessing and training script
├── visualize.py         # Plots the confusion matrices saved during training
├── evaluate.py          # Tests of the numbered TODOs + accuracy check of your weights
├── model_tournament.py  # (optional) the model and the preprocessing of the tournament
├── train_job.sbatch     # SLURM job script for cluster training
├── train_job_4gpu.sbatch # the same with 4 GPUs
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

> 💡 **Note:** For cluster details, see the [Course Page](https://cw.fel.cvut.cz/wiki/courses/becm33dpl/resources/coding)

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone git@github.com:jskvrna/DPL-HW3.git
```

### 2️⃣ Set up your environment
Use **Python 3.10** (recommended; the pinned versions in `requirements.txt` also work with 3.11–3.13, but not with 3.14). Create a virtual environment and install dependencies:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Get the dataset
There are two versions of the dataset (**fungi40c**) with the same structure:

| Version | Size | Where | Use it for |
|---------|------|-------|------------|
| **fungi_tiny** | 600 images (≤5 per class in each split), ≈80 MB | Download from the [Google Drive folder](https://drive.google.com/drive/folders/1OLMsrB_Z6cJAyzgFOsbLzDZMLgRK-YLI?usp=sharing), or copy it from the CTU machines with `rsync -avP <your_username>@ptak.felk.cvut.cz:/mnt/datasets/dpl/fungi_tiny .` | Writing and debugging your code **on your own computer** |
| **fungi40c** (full) | ≈61.5k images, **≈10 GB** | `/mnt/datasets/dpl/fungi40c` on the CTU machines and the GPU cluster, or from the same [Google Drive folder](https://drive.google.com/drive/folders/1OLMsrB_Z6cJAyzgFOsbLzDZMLgRK-YLI?usp=sharing) | The real training, **on the GPU cluster** |

Recommended workflow:
1. Download `fungi_tiny`, unpack it anywhere on your computer and set `data_dir` in `confs/config.yml` to the unpacked folder (the one containing `train/`, `val/`, `test/`). Use an absolute path: `~` is **not** expanded (write `/home/<you>/fungi_tiny`, not `~/fungi_tiny`), otherwise the dataset is not found.
2. Fill in the templates and check that `python train.py` runs end-to-end on it (a few epochs are enough, the accuracy will be low). Then run the **overfit test** (see step 5) to see that your model really learns.
3. Copy your code to the GPU cluster, set `data_dir` to `/mnt/datasets/dpl/fungi40c` and train there (see step 6 below). You can download the full dataset to your computer too, but it is ≈10 GB and training from scratch on it is slow without a GPU, so use the cluster.

The dataset has 40 classes (folders `00` ... `39`) and is split into `train/` (≈59.5k images in the full version), `val/` (1000) and `test/` (1000):
```
<data_dir>/
├── train/00/*.jpg ... train/39/*.jpg
├── val/00/*.jpg   ... val/39/*.jpg
└── test/00/*.jpg  ... test/39/*.jpg
```
Images are 300 px wide with varying height; the class is given by the folder name. The final grading is done on a **private** test set with the same classes. Note that the classes are imbalanced.

### 4️⃣ Configure your experiment
Set the hyperparameters in `confs/config.yml` (TODO 4.2).

### 5️⃣ Train the model
- Fill in the TODOs in `train.py` and `model.py` (look for `### TODO x.y ###`, the numbers are listed in [Your Tasks](#-your-tasks)). Go through the sections 1 to 6 in this order, each one builds on the previous ones. Everything else (the per-channel statistics `MEAN`/`STD`, device selection, logging, the validation metrics, `main()`) is given. You have to use the provided `train.py`, you cannot replace it by your own script.
- **Tests first:** `python evaluate.py` checks every numbered TODO and prints `[PASS]`/`[FAIL]` for it, see [Your Tasks](#-your-tasks). Run it after each step and make sure that everything passes **before you start any training**.
- **Overfit test (then, on `fungi_tiny`):** `python train.py --overfit N` trains **and** validates on the same `N` random training images (e.g. `N` = 64), without augmentation. A correct pipeline (dataset, labels, model, loss, optimizer, `zero_grad`, train mode, ...) memorizes them: the training loss goes to ≈0 and the accuracy to 100% within a few dozen epochs (set `num_epochs` accordingly, it takes seconds). If it does not, something is broken: fix it before you spend hours on the full dataset. Take `N` at least as large as `batch_size`, otherwise a batch is not even full.
- **Train:** run the training script
```bash
python train.py
```
- The loss, accuracy, [macro F1](https://en.wikipedia.org/wiki/F-score) (the F1 score averaged over the classes, see also [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)) and [mAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) (the average precision of every class computed one-vs-rest from the softmax scores, averaged over the classes, see also [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)) of every validation are printed to the console and saved to `experiments/<experiment>/logs/experiment.log`.
- **[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix):** every validation also saves the confusion matrix (rows = true class, columns = predicted class) to `experiments/<experiment>/confusion/epoch_XXX.npy`. Plot it with `python visualize.py experiments/<experiment>/confusion/epoch_010.npy` (writes a `.png` next to it and prints the most frequent mistakes; needs `matplotlib`, e.g. on your own computer): which species does your model mix up?
- **(Optional) Weights & Biases:** to get nice live curves, create a free account at [wandb.ai](https://wandb.ai), run `wandb login` once and train with `python train.py --wandb`. The loss, accuracy, macro F1, mAP and the confusion matrix are logged to the project `dpl-hw3-fungi` (this needs `matplotlib`).
- The model is saved to `experiments/<experiment>/models/best_model.pth` every time the **validation accuracy** reaches a new maximum (the best model is chosen by the accuracy only, not by the loss, macro F1 or mAP, and it is not the model of the last epoch). Only the weights (`state_dict`) are saved, in the precision of the model (float32).

### 6️⃣ Train on the GPU cluster (SLURM)
Training from scratch on ~60k images is slow on a CPU, so use a GPU node:
```bash
cd <path to this repository>      # on the cluster, e.g. in your home
mkdir -p logs                     # SLURM does not create the output directory, the job fails without it
sbatch train_job.sbatch           # submit; prints the job ID
squeue -u $USER                   # your jobs (PD = waiting, R = running)
tail -f logs/trn-<jobid>.out      # follow the training log
scancel <jobid>                   # cancel the job
```
- The job script requests 1 GPU, 8 CPU cores, 32 GB RAM and 24 hours. Set `num_workers` in the config to at most the number of CPU cores per GPU (8), otherwise data loading will be the bottleneck.
- **Several GPUs (optional):** request them with `#SBATCH --gres=gpu:2` in `train_job.sbatch`. `train.py` then splits every batch between the GPUs automatically (`nn.DataParallel`, the `batch_size` and `num_workers` from the config are used **per GPU**, so with 4 GPUs the real batch size and the number of workers are 4× larger; the saved weights are the same as with one GPU). `train_job_4gpu.sbatch` is a ready-made script with 4 GPUs and 32 CPU cores (8 per GPU): submit it with `sbatch train_job_4gpu.sbatch`. The speedup is real only if the data loading keeps up: raise `--cpus-per-task` proportionally to the number of GPUs (e.g. 16 for 2 GPUs).
- Your job can be killed when the time limit is over, so test on a few epochs first and estimate how long an epoch takes before you start a long run.
- To get your results back, copy `experiments/` to your computer with `rsync -avP`.
- **(Optional) Weights & Biases on the cluster:** if `wandb` is missing in the module environment, install it with `pip install --user wandb`, run `wandb login` once and enable W&B in `train_job.sbatch` by changing its last line to `python train.py --wandb`. If the compute nodes have no internet access, set `WANDB_MODE=offline` in the job script and upload the run afterwards with `wandb sync` from a machine with internet.

---

## ✅ Your Tasks

Every place to fill in is marked `### TODO x.y ###` in the code. Implement all of them:

**1. 📂 Dataset** (`FungiDataset(root, split, train)` in `train.py`; the flag `train` says if it is the training set (with augmentation) or not)
- **1.1** `classes`: sorted list of the class folder names (the label of a class is its index in this list)
- **1.2** `samples`: list of `(image path, label)` for every image
- **1.3** `__len__`: number of samples
- **1.4** `__getitem__`: load the image as RGB, preprocess it with `get_train_transform()` if `self.train` is set and with `get_eval_transform()` otherwise, return `(image, label)`

**2. 🖼️ Preprocessing** (`train.py`, uses the given `MEAN` and `STD`)
- **2.1** `get_train_transform()`: resize + center crop to 224×224, augmentations of your choice (they regularize: the model sees a different image in every epoch), `ToTensor`, `Normalize(MEAN, STD)` (`MEAN` and `STD`, the per-channel statistics of the train set, are given in `train.py`)
- **2.2** `get_eval_transform()`: the same without augmentations. **The grader imports and calls this function** (without arguments) to preprocess the test images before they go into your model, so it must work standalone. It receives a PIL image already converted to RGB (`Image.open(path).convert("RGB")`).

**3. 🏗️ Model** (`model.py`)
- **3.1** the network in `MyModel`: a CNN built as `nn.Sequential` (trained from scratch), with a regularization such as `nn.Dropout`. `MyModel` accepts `output_size` (40 classes); the input is a normalized batch `[batch_size, 3, 224, 224]` and the output are class logits `[batch_size, 40]`.

**4. 📦 Data loading and configuration** (`create_dataloaders` in `train.py`, `confs/config.yml`)
- **4.1** the train and validation `DataLoader`s (batch size and number of workers from the config, shuffling only where it belongs)
- **4.2** `confs/config.yml`: replace the ❓ by values of `batch_size`, `learning_rate`, `num_epochs`, `num_workers` (at most the number of CPU cores) and `validate_every`. `batch_size` and `num_workers` are per GPU. Start with safe values, tune them in 8.

**5. 🔄 Training loop** (`run_training` in `train.py`)
- **5.1** loss function, optimizer (think about weight decay, e.g. AdamW) and (optionally) a learning rate scheduler
- **5.2** switch the model to training mode
- **5.3** move the images and labels to the device
- **5.4** reset the gradients
- **5.5** make predictions and compute the loss
- **5.6** backpropagate and make an optimizer step
- **5.7** count the correct predictions (training accuracy)

**6. ✔️ Validation** (`evaluate` in `train.py`)
- **6.1** run without gradient tracking
- **6.2** switch the model to evaluation mode
- **6.3** make predictions and compute the loss

**7. 🏋️ Overfit test** (no code to fill in, do it **before** you spend time on 8): on `fungi_tiny`, `python train.py --overfit N` (e.g. `N` = 64) must reach ≈0 training loss and 100% accuracy (see [step 5](#5%EF%B8%8F⃣-train-the-model)). If it does not, find the bug before you continue.

**8. ⚙️ Tune and train** (no code to fill in)
- tune the values you chose in 4.2 (and your model, augmentations, optimizer, ...) for optimal performance (monitor the log file or, optionally, Weights & Biases)
- train on the full dataset on the GPU cluster (see [step 6](#6%EF%B8%8F⃣-train-on-the-gpu-cluster-slurm)), pick the best run by validation accuracy and submit its `best_model.pth` as `weights.pth`.

**9. 🏆 Tournament (optional, see [Tournament](#-tournament-additional-part))**
- **9.1** `MyModel` in `model_tournament.py`: your tournament model (pretrained backbones are allowed)
- **9.2** `get_eval_transform_tournament()` in `model_tournament.py`: its preprocessing (optional, a default is given)

### 🧪 Check your work: `python evaluate.py`
`evaluate.py` first runs a test for every numbered TODO (19 tests) on small synthetic data (no dataset needed) and prints which ones pass:
```
[PASS] 1.1 FungiDataset.classes: sorted class folder names
[FAIL] 1.4 FungiDataset.__getitem__: RGB image tensor + label (also for grayscale / RGBA files)
         -> AssertionError: expected shape (3, 224, 224), got (1, 224, 224) (convert every image to RGB)
```
- Item 5.3 can only fail when you train on a GPU (or Apple MPS); on a CPU it always passes.
- The tests expect the structure of the template (function names and arguments), do not rename or re-sign the given functions.
- The tests are a help to find bugs, the grade depends on the accuracy below.

---

## 📊 Evaluation

### 🎯 Basic Evaluation (10 points)

The model is evaluated by **accuracy** on a private test set (the share of correctly classified images):

| Accuracy | Points |
|----------|--------|
| ≥ 45% | 4 |
| ≥ 55% | 7 |
| ≥ 65% | 10 |

Before submitting, check your weights with the grader's interface:
```bash
python evaluate.py --data-dir <data_dir> --weights weights.pth    # tests + accuracy, macro F1, mAP, the combined score (their mean) and a confusion matrix plot on <data_dir>/val
```

## 🏆 Tournament (additional part)

On top of the basic part there is a tournament: your model competes with the models of the other students, ranked by its **score** on the private test set: the average of the **accuracy**, the **macro F1** and the **mAP**. The tournament is **voluntary**, so everything about it (part 9 below, `model_tournament.py`, `weights_tournament.pth`) is optional. It is a **separate submission**, independent of the homework zip.

**Points** (bonus points on top of the 10 points of the basic part):
- every valid submission: **+1**, the better half of the ranking: **+2**, the best 25 %: **+3**
- the winner gets **+2** more, the second place **+1** more

**Deadline: 2 December 2026, 23:59**

- **Pretrained models and transfer learning are allowed** here, unlike in the basic part. Use whatever works best.
- **Separate files**, so the basic part stays from scratch: `model_tournament.py` and `weights_tournament.pth`. `model_tournament.py` defines the class `MyModel(output_size=40)` and the function `get_eval_transform_tournament()`; it does not depend on `train.py` or `model.py`. Train the model with your own training script (e.g. a modified copy of `train.py`) and its own `confs/config.yml`: both are **submitted with the tournament** (the ones that trained your `weights_tournament.pth`). Create the model without pretrained weights in `MyModel` and load the pretrained weights of your backbone in your training script (this needs internet: do it on your computer).
- **Preprocessing:** the grader calls `get_eval_transform_tournament()` from `model_tournament.py` (a default with the ImageNet mean/std is given). Change it if your model needs a different preprocessing, e.g. another resolution. The model takes the batch made by this transform and returns `[batch_size, 40]` class logits.
- **10 submissions.** You can submit up to 10 times before the deadline. Only your **best** submission (the highest score) counts in the ranking.
- **Size limit: the tournament zip must be at most 500 MB** (the same limit as for the homework, counted separately: the two submissions do not share it). Check the size of your weights before you submit; if they are too large, save them in half precision, see [Submission](#-submission).
- The model **must not download anything** when it is created or run (there is no internet access during the evaluation): the weights of the pretrained backbone come from `weights_tournament.pth`, not from the internet.
- Your code is run in a sandbox without internet access and without access to the labels of the private test set.
- **Check your files:** `python evaluate.py --tournament` tests items 9.1 and 9.2 of `model_tournament.py` (the model is built without internet and gives `[batch_size, 40]` logits, the transform works), and `python evaluate.py --tournament --data-dir <data_dir> --weights weights_tournament.pth` also prints the accuracy, macro F1, mAP and score of your tournament model on `<data_dir>/val` and the size of the weights file.

## 📦 Submission

There are **two separate submissions**, each is a **zip file** of at most **500 MB**. The files must be **in the root of the zip** (e.g. `zip -r hw3.zip model.py train.py confs/config.yml weights.pth` and `zip -r tournament.zip model_tournament.py train.py confs/config.yml weights_tournament.pth`), not inside a folder: a zip of the whole repository folder fails with "files missing".

**1. The homework (HW3):**
- ✅ `model.py` - Your neural network architecture
- ✅ `train.py` - The `train.py` template filled in by you (the grader imports `get_eval_transform()` from it)
- ✅ `confs/config.yml` - Your hyperparameters (TODO 4.2)
- ✅ `weights.pth` - Trained model weights (state_dict; copy your `best_model.pth`)

**2. The tournament (optional, see [Tournament](#-tournament-additional-part)):**
- 🏆 `model_tournament.py` - Your tournament model and `get_eval_transform_tournament()`
- 🏆 `weights_tournament.pth` - Its weights (state_dict)
- 🏆 `train.py` - The training script of your tournament model (e.g. your modified copy of `train.py`)
- 🏆 `confs/config.yml` - Its hyperparameters

The weights are saved as a plain `state_dict` in float32. If they would exceed the limit, save them in half precision, `load_state_dict` converts them back when the grader loads them:
```python
torch.save({k: v.half() for k, v in model.state_dict().items()}, "weights.pth")
```

---

## 🚫 Important Policies

### 🤖 Pretrained Models Policy
❌ In the **basic part** (the accuracy thresholds above) you are **not allowed** to use any pretrained models or transfer learning techniques.  
✅ All models there **must be trained from scratch** using the provided dataset and the `train.py` that you submit.  
✅ In the 🏆 **tournament** (below) pretrained models and transfer learning **are allowed**.

### 📝 Plagiarism Policy
✅ You are **strongly encouraged** to discuss ideas and approaches with your peers.  
⚠️ However, the code you submit **must be your own work** and you must fully understand it.
❌ Copying code from others is **strictly prohibited** and will result in a **zero score** for the assignment.

---

<div align="center">
  
**Good luck! 🍀**

**In case of any questions, feel free to write me at jan.skvrna@cvut.cz**

</div>
