"""Check your solution the way the grader does.

1. Tests of the numbered TODOs in train.py and model.py (1.x dataset, 2.x preprocessing, 3.x model, 4.x data loaders,
   5.x training loop, 6.x validation) on small synthetic data. They tell you which TODO is still wrong.
2. If `--weights` is given: accuracy of your weights on a split of the real dataset.

    python evaluate.py                                                # only the tests
    python evaluate.py --data-dir <data_dir> --weights weights.pth    # tests + accuracy on <data_dir>/val
"""
import argparse
import contextlib
import socket
import logging
import os
import re
import tempfile

os.environ["TQDM_DISABLE"] = "1"  # no progress bars in the test output

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import train
from model import MyModel

NUM_CLASSES = 40
TESTS = []  # (item, description, function) of the basic part
TOURNAMENT_TESTS = []  # the same for the optional tournament files (python evaluate.py --tournament)
CTX = {}  # shared test fixtures: "root" (synthetic dataset), "config_path"


def test(item, description, tournament=False):
    def register(fn):
        (TOURNAMENT_TESTS if tournament else TESTS).append((item, description, fn))
        return fn
    return register


# ---------------------------------------------------------------- fixtures
def make_root(root):
    """Tiny synthetic dataset with a grayscale JPEG and an RGBA PNG: <root>/val/<00|01|02>/*."""
    rng = np.random.default_rng(0)

    def rgb(w, h):
        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    for c in ("00", "01", "02"):
        os.makedirs(os.path.join(root, "val", c))
    rgb(300, 225).convert("L").save(os.path.join(root, "val", "00", "gray.jpg"))
    rgb(300, 400).convert("RGBA").save(os.path.join(root, "val", "00", "rgba.png"))
    rgb(300, 225).save(os.path.join(root, "val", "01", "a.jpg"))
    rgb(300, 169).save(os.path.join(root, "val", "01", "b.jpg"))
    rgb(300, 400).save(os.path.join(root, "val", "02", "c.jpg"))


def load_config():
    with open(CTX["config_path"]) as f:
        return yaml.safe_load(f)


PLAIN = transforms.Compose([transforms.Resize(train.IMAGE_SIZE), transforms.CenterCrop(train.IMAGE_SIZE),
                            transforms.ToTensor()])


def solid(w, h, color=(255, 255, 255)):
    return Image.new("RGB", (w, h), color)


class Spy(nn.Module):
    """Tiny model that records how it is called. kind="const": always predicts class 0 (logit 3 vs 0)."""

    def __init__(self, kind="linear"):
        super().__init__()
        self.kind = kind
        self.fc = nn.Linear(4, 2)
        self.calls = []

    def forward(self, x):
        stale_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.parameters())
        self.calls.append(dict(bs=len(x), training=self.training, grad_enabled=torch.is_grad_enabled(),
                               stale_grad=stale_grad))
        if self.kind == "const":
            logits = torch.zeros(len(x), 2, device=x.device)
            logits[:, 0] = 3.0
            return logits + 0 * self.fc(x)
        return self.fc(x)


def toy_loaders():
    """Train: 8 samples, batch 4. Val: 6 samples, batch 3. Labels alternate 0, 1, 0, 1, ..."""
    g = torch.Generator().manual_seed(0)

    def ds(n):
        y = torch.arange(n) % 2
        return TensorDataset((y * 2 - 1)[:, None].float() + 0.1 * torch.randn(n, 4, generator=g), y)

    return (DataLoader(ds(8), batch_size=4, shuffle=True, generator=g), DataLoader(ds(6), batch_size=3))


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run_training_on_toy(kind, epochs, validate_every):
    """Run train.run_training on the toy data. Returns (model, initial weights, log lines, best val acc)."""
    device = train.get_device()
    model = Spy(kind).to(device)
    init = [p.detach().clone() for p in model.parameters()]
    # fixed values (not the student's config, so that 5.x does not depend on 4.2)
    config = {**load_config(), "learning_rate": 0.01, "num_epochs": epochs, "validate_every": validate_every,
              "output_dim": 2}
    logger = logging.getLogger("evaluate_tests")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = ListHandler()
    logger.handlers = [handler]
    train_loader, val_loader = toy_loaders()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "models"))
        best = train.run_training(train_loader, val_loader, config, tmp, logger, model, None, device)
    return model, init, handler.lines, best


def logged(lines, key):
    return [float(m.group(1)) for line in lines if (m := re.search(rf"{key}: ([0-9.]+)", line))]


TRAIN_RUNS = {}


def training_run(name):
    """Run each toy training only once, reuse it (or its exception) for all the 5.x tests."""
    if name not in TRAIN_RUNS:
        try:
            TRAIN_RUNS[name] = run_training_on_toy(*{"learn": ("linear", 30, 10), "const": ("const", 1, 1)}[name])
        except Exception as e:
            TRAIN_RUNS[name] = e
    if isinstance(TRAIN_RUNS[name], Exception):
        raise AssertionError(f"run_training crashed (see 5.1): {type(TRAIN_RUNS[name]).__name__}: {TRAIN_RUNS[name]}")
    return TRAIN_RUNS[name]


# ---------------------------------------------------------------- 1.x dataset
@contextlib.contextmanager
def plain_transforms():
    """Replace get_train_transform / get_eval_transform by PLAIN, so that 1.x does not depend on 2.x."""
    orig = train.get_train_transform, train.get_eval_transform
    train.get_train_transform = train.get_eval_transform = lambda: PLAIN
    try:
        yield
    finally:
        train.get_train_transform, train.get_eval_transform = orig


def dataset(train_flag=False):
    return train.FungiDataset(CTX["root"], "val", train_flag)


@test("1.1", "FungiDataset.classes: sorted class folder names")
def _():
    classes = list(dataset().classes)
    assert classes == ["00", "01", "02"], f"expected ['00', '01', '02'], got {classes}"


@test("1.2", "FungiDataset.samples: (path, label) for every image")
def _():
    ds = dataset()
    assert len(ds.samples) == 5, f"expected 5 samples, got {len(ds.samples)}"
    for path, label in ds.samples:
        assert os.path.isfile(path), f"{path} is not a file"
        assert isinstance(label, (int, np.integer)), f"label must be an int, got {type(label).__name__}"
        assert ds.classes[label] == os.path.basename(os.path.dirname(path)), f"wrong label {label} for {path}"


@test("1.3", "FungiDataset.__len__")
def _():
    assert len(dataset()) == 5, f"expected 5, got {len(dataset())}"


@test("1.4", "FungiDataset.__getitem__: RGB image tensor + label (also for grayscale / RGBA files)")
def _():
    for flag in (True, False):
        ds = dataset(flag)
        for i, (path, label) in enumerate(ds.samples):
            with plain_transforms():
                image, y = ds[i]
            name = os.path.basename(path)
            assert isinstance(image, torch.Tensor), f"image must be a tensor, got {type(image).__name__}"
            assert image.shape == (3, 224, 224), f"{name}: expected shape (3, 224, 224), got {tuple(image.shape)} " \
                                                 f"(convert every image to RGB)"
            assert y == label, f"{name}: expected label {label}, got {y}"


# ---------------------------------------------------------------- 2.x preprocessing
@test("2.1", "get_train_transform: PIL image -> float tensor [3, 224, 224]")
def _():
    tf = train.get_train_transform()
    for size in ((300, 225), (300, 400)):
        out = tf(Image.fromarray(np.random.default_rng(0).integers(0, 256, size[::-1] + (3,), dtype=np.uint8)))
        assert isinstance(out, torch.Tensor) and out.dtype == torch.float32, "output must be a float32 tensor"
        assert out.shape == (3, 224, 224), f"image {size}: expected shape (3, 224, 224), got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "output contains NaN / inf"


@test("2.2", "get_eval_transform: deterministic, normalized, resize + center crop")
def _():
    tf = train.get_eval_transform()
    mean, std = (torch.as_tensor(v, dtype=torch.float32)[:, None, None] for v in (train.MEAN, train.STD))
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (225, 300, 3), dtype=np.uint8))
    a, b = tf(img), tf(img)
    assert a.shape == (3, 224, 224), f"expected shape (3, 224, 224), got {tuple(a.shape)}"
    assert torch.equal(a, b), "the eval transform must be deterministic (no random augmentation)"
    white = tf(solid(300, 225))
    assert torch.allclose(white, ((1 - mean) / std).expand(3, 224, 224), atol=1e-3), \
        "a white image should become (1 - MEAN) / STD: is Normalize(MEAN, STD) the last step?"
    border = solid(300, 225)
    border.paste((255, 0, 0), (0, 0, 10, 225))  # red 10 px stripe on the left edge
    assert torch.allclose(tf(border), white, atol=1e-3), \
        "the image is squashed instead of cropped: resize the shorter side to 224, then center crop"
    step = solid(600, 450)  # black for x < 200 of 600, i.e. ~28 % of the center crop after resizing
    step.paste((0, 0, 0), (0, 0, 200, 450))
    black = (tf(step)[0, 112] < ((0.5 - mean[0]) / std[0])).float().mean().item()
    assert 0.15 < black < 0.45, f"expected ~28 % of the row to be black, got {black:.0%}: is the image resized first?"


# ---------------------------------------------------------------- 3.x model
@test("3.1", "MyModel: nn.Sequential CNN, [B, 3, 224, 224] -> [B, 40] logits")
def _():
    model = MyModel(output_size=NUM_CLASSES)
    assert isinstance(model.net, nn.Sequential) and len(model.net) > 0, "model.net must be a non-empty nn.Sequential"
    assert any(isinstance(m, nn.Conv2d) for m in model.modules()), "a CNN needs convolutional layers"
    model.train()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES), f"expected output shape (2, {NUM_CLASSES}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "output contains NaN / inf"
    assert (out < 0).any() or not torch.allclose(out.sum(1), torch.ones(2), atol=1e-3), \
        "the output must be logits (no softmax, the loss applies it)"
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters()), "some parameters get no gradient (unused layers?)"
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, NUM_CLASSES), "batch of 1 must work in eval mode"


# ---------------------------------------------------------------- 4.x data loaders
@test("4.1", "create_dataloaders: batch_size / num_workers from config, shuffle only for train")
def _():
    config = {**load_config(), "batch_size": 3, "num_workers": 1}  # not the student's values: independent of 4.2
    train_loader, val_loader = train.create_dataloaders(*(l.dataset for l in toy_loaders()), config)
    assert isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader), "must return two DataLoaders"
    for name, loader in (("train", train_loader), ("val", val_loader)):
        assert loader.batch_size == config["batch_size"], f"{name}: batch_size must be config['batch_size']"
        assert loader.num_workers == config["num_workers"], f"{name}: num_workers must be config['num_workers']"
    assert isinstance(train_loader.sampler, RandomSampler), "the train loader must shuffle"
    assert isinstance(val_loader.sampler, SequentialSampler), "the validation loader must not shuffle"


@test("4.2", "config.yml: batch_size, learning_rate, num_epochs, num_workers, validate_every are set")
def _():
    config = load_config()
    for key, low in (("batch_size", 1), ("num_epochs", 1), ("num_workers", 0), ("validate_every", 1)):
        v = config[key]
        assert isinstance(v, int) and not isinstance(v, bool), f"{key} must be an integer, got {v!r} (replace the ❓)"
        assert v >= low, f"{key} must be at least {low}, got {v}"
    lr = config["learning_rate"]
    assert isinstance(lr, float) and 0 < lr < 1, f"learning_rate must be a number in (0, 1), got {lr!r} (replace the ❓)"
    assert config["validate_every"] <= config["num_epochs"], \
        "validate_every > num_epochs: the model would never be validated and saved"


# ---------------------------------------------------------------- 5.x training loop (train.run_training)
@test("5.1", "run_training: loss function and optimizer defined, runs without error")
def _():
    training_run("learn")


@test("5.2", "run_training: model in train mode in every epoch (also after validation)")
def _():
    calls = [c for c in training_run("learn")[0].calls if c["bs"] == 4]
    assert calls and all(c["training"] for c in calls), \
        "model.train() must be called at the start of every epoch (validation switches to eval mode)"


@test("5.3", "run_training: images and labels moved to the device (only checked on GPU / MPS)")
def _():
    training_run("learn")  # a missing .to(device) crashes there when the model is on a GPU


@test("5.4", "run_training: gradients zeroed before every step")
def _():
    calls = [c for c in training_run("learn")[0].calls if c["bs"] == 4]
    assert not any(c["stale_grad"] for c in calls), "gradients of the previous batch are still there: zero_grad()"


@test("5.5", "run_training: loss decreases on an easy toy problem")
def _():
    losses = logged(training_run("learn")[2], "Train Loss")
    assert losses[-1] < losses[0], f"train loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@test("5.6", "run_training: backward pass and optimizer step update the weights")
def _():
    model, init, _, _ = training_run("learn")
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), init)), "the weights did not change"


@test("5.7", "run_training: training accuracy counted correctly")
def _():
    acc = logged(training_run("const")[2], "Train Accuracy")[0]
    assert abs(acc - 0.5) < 1e-3, f"a model that always predicts class 0 on balanced data has accuracy 0.5, got {acc}"


# ---------------------------------------------------------------- 6.x validation (train.evaluate)
def run_evaluate():
    if "evaluate" not in TRAIN_RUNS:
        device = train.get_device()
        model = Spy("const").to(device).train()
        try:
            out = train.evaluate(model, toy_loaders()[1], nn.CrossEntropyLoss(), device, 2)
            TRAIN_RUNS["evaluate"] = (model, out)
        except Exception as e:
            TRAIN_RUNS["evaluate"] = e
    if isinstance(TRAIN_RUNS["evaluate"], Exception):
        raise AssertionError(f"evaluate crashed: {type(TRAIN_RUNS['evaluate']).__name__}: {TRAIN_RUNS['evaluate']}")
    return TRAIN_RUNS["evaluate"]


@test("6.1", "evaluate: runs without gradient tracking")
def _():
    assert not any(c["grad_enabled"] for c in run_evaluate()[0].calls), "gradients are tracked during evaluation"


@test("6.2", "evaluate: model in evaluation mode")
def _():
    assert not any(c["training"] for c in run_evaluate()[0].calls), "the model is still in train mode"


@test("6.3", "evaluate: predictions, loss and accuracy")
def _():
    out = run_evaluate()[1]
    loss, acc = out["loss"], out["acc"]
    logits = torch.tensor([[3.0, 0.0]]).repeat(6, 1)
    expected = nn.CrossEntropyLoss()(logits, torch.arange(6) % 2).item()
    assert abs(loss - expected) < 1e-3, f"expected loss {expected:.4f}, got {loss:.4f}"
    assert abs(acc - 0.5) < 1e-3, f"expected accuracy 0.5, got {acc}"


# ---------------------------------------------------------------- 9.x tournament (python evaluate.py --tournament)
@contextlib.contextmanager
def no_network():
    """The evaluation has no internet: make every connection (and DNS lookup) fail."""
    def blocked(*args, **kwargs):
        raise OSError("no internet during the evaluation: do not download anything when the model is created")
    connect, getaddrinfo = socket.socket.connect, socket.getaddrinfo
    socket.socket.connect, socket.getaddrinfo = blocked, blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.getaddrinfo = connect, getaddrinfo


@test("9.1", "model_tournament.MyModel: built without internet, transform output -> [B, 40] logits, "
             "all weights in the state_dict", tournament=True)
def _():
    import model_tournament
    with no_network():
        model = model_tournament.MyModel(output_size=NUM_CLASSES)
        other = model_tournament.MyModel(output_size=NUM_CLASSES)
    assert isinstance(model, nn.Module), "MyModel must be an nn.Module"
    other.load_state_dict(model.state_dict())
    tf = model_tournament.get_eval_transform_tournament()
    batch = torch.stack([tf(solid(300, 400, (200, 50, 50))), tf(solid(300, 225))])
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, NUM_CLASSES), f"expected logits of shape (2, {NUM_CLASSES}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "the logits contain NaN / inf"


@test("9.2", "get_eval_transform_tournament: PIL image -> float tensor [3, H, W], deterministic", tournament=True)
def _():
    import model_tournament
    tf = model_tournament.get_eval_transform_tournament()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (400, 300, 3), dtype=np.uint8))
    a, b = tf(image), tf(image)
    assert isinstance(a, torch.Tensor) and a.dtype == torch.float32 and a.dim() == 3 and a.shape[0] == 3, \
        f"expected a float32 tensor [3, H, W], got {getattr(a, 'shape', type(a))}"
    assert torch.equal(a, b), "the transform must be deterministic (no random augmentation)"
    assert torch.isfinite(a).all(), "output contains NaN / inf"


# ---------------------------------------------------------------- main
def run_tests(tests):
    passed = 0
    for item, description, fn in tests:
        try:
            fn()
            status, msg = "PASS", ""
            passed += 1
        except Exception as e:
            text = f"{type(e).__name__}: {e}"
            status = "FAIL"
            msg = "not implemented yet" if isinstance(e, NotImplementedError) or "ellipsis" in text.lower() else text
        print(f"[{status}] {item} {description}" + (f"\n         -> {msg}" if msg else ""))
    print(f"\nTests passed: {passed}/{len(tests)}")
    return passed == len(tests)


def compute_accuracy(args):
    device = train.get_device()
    if args.tournament:
        from model_tournament import MyModel as Model, get_eval_transform_tournament
        transform = get_eval_transform_tournament()
    else:
        Model, transform = MyModel, train.get_eval_transform()
    print(f"Size of {args.weights}: {os.path.getsize(args.weights) / 2 ** 20:.1f} MB")
    model = Model(output_size=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=torch.device("cpu")))
    model.to(device).eval()

    dataset = ImageFolder(os.path.join(args.data_dir, args.split), transform=transform)
    assert len(dataset.classes) == NUM_CLASSES
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

    scores, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            scores.append(model(images.to(device)).softmax(dim=1).cpu())
            labels.append(y)
    scores, labels = torch.cat(scores), torch.cat(labels)
    preds = scores.argmax(dim=1)
    print(f"Accuracy on {args.split}: {(preds == labels).float().mean().item():.4f}")
    accuracy = (preds == labels).float().mean().item()
    f1 = train.macro_f1(train.confusion_matrix(preds, labels, NUM_CLASSES))
    mean_ap = train.mean_average_precision(scores, labels, NUM_CLASSES)
    print(f"Macro F1 on {args.split}: {f1:.4f}")
    print(f"mAP on {args.split}: {mean_ap:.4f}")
    print(f"Score on {args.split} (mean of accuracy, macro F1 and mAP): {(accuracy + f1 + mean_ap) / 3:.4f}")

    import visualize  # needs matplotlib
    confusion = train.confusion_matrix(preds, labels, NUM_CLASSES).numpy()
    png = f"confusion_{args.split}.png"
    visualize.save_confusion(confusion, png, title=args.split)
    print(f"Confusion matrix saved to {png}, the most frequent mistakes:")
    for true, pred, count in visualize.top_confusions(confusion, 5):
        print(f"    class {true:02d} predicted as {pred:02d}: {count}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="confs/config.yml")
    parser.add_argument("--tournament", action="store_true",
                        help="check the tournament file (model_tournament.py: model and get_eval_transform_tournament) "
                             "instead of the basic ones; use it with --weights weights_tournament.pth")
    parser.add_argument("--weights", help="weights.pth: also compute the accuracy on <data-dir>/<split>")
    parser.add_argument("--data-dir", help="dataset folder with train/, val/, test/ (needed with --weights)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.weights and not args.data_dir:
        parser.error("--weights needs --data-dir")

    with tempfile.TemporaryDirectory() as root:
        make_root(root)
        CTX.update(root=root, config_path=args.config)
        all_passed = run_tests(TOURNAMENT_TESTS if args.tournament else TESTS)

    if args.weights:
        if not all_passed:
            print("\nWARNING: some tests failed, the accuracy below may be wrong or the evaluation may crash.")
        compute_accuracy(args)


if __name__ == "__main__":
    main()
