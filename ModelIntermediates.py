import torch
from torch import nn
import torchvision.models as models

def list_intermediates(model, verbose=True):
    names = dict(model.named_children())
    if verbose:
        for name, layer in names.items():
            print(name, '--', layer)
    return list(names.keys())

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self._model = models.vgg16(pretrained=True).features.eval()
        self.layer_names = ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
                            "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
                            "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3","maxpool3",
                            "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3","maxpool4",
                            "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3","maxpool5"]

    def children(self):
        return self._model.children()

    def forward(self, x):
        return self._model(x)

class ModelIntermediates(nn.Module):
    def __init__(self, model, requested_names):
        super(ModelIntermediates, self).__init__()
        self._model = model
        self.intermediates = {}
        self._register_hooks(requested_names)

    def _register_hooks(self, requested_names):
        layer_names = self._model.layer_names #list_intermediates(self._model, verbose=False)
        requested_layers = [layer_names.index(n) for n in requested_names]
        for i, m in enumerate(self._model.children()):
            if isinstance(m, nn.ReLU):   # we want to replace the relu functions with in place functions.
                m.inplace = False        # the model has a hard time going backwards on the in place functions.
            if i in requested_layers:
                def curry(i):
                    def hook(module, input, output):
                        self.intermediates[layer_names[i]] = output
                    return hook
                m.register_forward_hook(curry(i))

    def forward(self, x):
        self._model(x)
        return self.intermediates
