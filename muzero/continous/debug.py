import logging
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
from torchvision.transforms.v2 import ToImage, ToDtype, Compose
from typing import Optional

class NoMatplotFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return not record.name.startswith("matplotlib")
    

def debug_plot(x: torch.Tensor):
    import matplotlib.pyplot as plt

    if x.dim() == 4:
        print("DEBUG: 4D tensor, taking first entry in the batch.")
        x = x[0]
    if x.shape[0] > 3:
        print("DEBUG: more than 3 channels, taking the first channel.")
        x = x[0]
    elif x.shape[0] == 2:
        print("DEBUG: 2 channels, taking the first channel.")
        x = x[0]
    plt.imshow(x)
    plt.show()


def plot_grad_flow(model: nn.Module, as_image: bool = False) -> Optional[torch.Tensor]:
    import matplotlib

    if as_image:
        matplotlib.use('Agg')  # This needs to be done before importing pyplot
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    named_parameters = model.named_parameters()
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        try:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                max_grads.append(p.grad.abs().max().cpu().detach().numpy())
        except:
            print("grad plot error in ", n)

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
        ['max-gradient', 'mean-gradient', 'zero-gradient'],
    )
    if as_image:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True)
        ])
        # Log the plot as an image in TensorBoard
        image = Image.open(buf)
        image = transform(image).unsqueeze(0)
        return image
    else:
        plt.show()