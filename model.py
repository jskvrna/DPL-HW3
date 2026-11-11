import torch.nn as nn


class MyModel(nn.Module):
    """CNN image classifier for the 40 fungi classes, trained from scratch.

    Input:  float tensor [batch_size, 3, 224, 224], already normalized (output of `get_eval_transform` in train.py).
    Output: class logits [batch_size, output_size] (no softmax, the loss takes care of it).
    """

    def __init__(self, output_size=40):
        """Initialize the model.

        Args:
            output_size (int): Number of classes (40).
        """
        super().__init__()
        self.net = nn.Sequential(
            ### TODO 3.1 ###
            # Define the whole network as a list of layers.
            # Think about: the order of Conv / BatchNorm / ReLU / pooling, how the number of channels and the
            # spatial size change from layer to layer, and how to get from a feature map to `output_size` logits.
            # Regularization: a small dataset overfits easily, think about where `nn.Dropout` belongs.
        )

    def forward(self, x):
        """Forward pass: [B, 3, 224, 224] -> [B, output_size] logits."""
        return self.net(x)
