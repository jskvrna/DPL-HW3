import torch.nn as nn
from torchvision import transforms


def get_eval_transform_tournament():
    """Preprocessing of the tournament model: the grader imports and calls this function (no arguments).

    It returns a callable: PIL image (already RGB) -> float tensor [3, H, W] that `MyModel` accepts. It must work
    standalone and be deterministic (no augmentation).
    """
    ### TODO 9.2 ###
    # Optional. The default below is resize + center crop to 224x224 with the ImageNet mean/std, which is what
    # torchvision's pretrained models expect. Change it if your model needs something else (another resolution, ...).
    return transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class MyModel(nn.Module):
    """Model for the tournament (optional part). Pretrained models and transfer learning are allowed here.

    Input:  the batch made by `get_eval_transform_tournament()`, [batch_size, 3, H, W].
    Output: class logits [batch_size, output_size] (no softmax).

    The grader creates the model as `MyModel(output_size=40)` WITHOUT internet access and then loads
    `weights_tournament.pth` into it, so never download anything in `__init__`.
    """

    def __init__(self, output_size=40):
        """Initialize the model.

        Args:
            output_size (int): Number of classes (40).
        """
        super().__init__()
        ### TODO 9.1 ###
        # Build your tournament model, e.g. a backbone from `torchvision.models` (or `timm`) with a new last layer for
        # `output_size` classes. Create it WITHOUT pretrained weights here (e.g. `weights=None`): load the pretrained
        # weights in your training script instead (there you have internet), train, and save the state_dict of the
        # whole model to `weights_tournament.pth` (check the 500 MB limit, saving in half precision halves the file).
        raise NotImplementedError

    def forward(self, x):
        """Forward pass: [B, 3, H, W] -> [B, output_size] logits."""
        raise NotImplementedError
