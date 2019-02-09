import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.nn.parameter import Parameter
from skimage.io import imread


def get_normalize(size):
    """
    Normalize Images
    """
    dim = 320
    normalize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    return (lambda x: normalize_transform(np.array(x)).unsqueeze(0))

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, inp):
        self.loss = F.mse_loss(inp, self.target)
        return self.loss

def gram_matrix(layer):
    # batch size is always 1
    layer = layer.squeeze()
    n,m,k = layer.size()
    layer = layer.reshape(n, m*k)

    gram_mat = torch.mm(layer, layer.t())
    return gram_mat/(m*k)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, inp):
        G = gram_matrix(inp)
        self.loss = F.mse_loss(G, self.target)
        return self.loss/4

def loss(loss_content, loss_style):
    alpha = 1
    beta = 1e4
    return alpha*loss_content + beta*loss_style

def plot_img(mat):
    mat = mat.detach()
    upper = np.percentile(mat,99)
    lower = np.percentile(mat, 1)
    mat = mat.clamp(lower, upper)
    mat = (mat-mat.min())/(mat.max()-mat.min())
    plt.imshow(np.rollaxis(np.array(mat.squeeze()), 0, 3))
    plt.axis('off')
    plt.show()

def style_transfer(model, content_image, style_image, requested_content, requested_style, size=240):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocess = get_normalize(size)
    content_image = preprocess(content_image).to(device)
    style_image = preprocess(style_image).to(device)

    gen_img = Parameter(torch.Tensor(content_image.size()))
    gen_img = gen_img.data.uniform_(0,1).to(device).requires_grad_()

    cont_layers = model(content_image)
    content_losses_f = [ContentLoss(layer).to(device) for layer in cont_layers.values()]
    style_layers = model(style_image)
    style_losses_f = [StyleLoss(layer).to(device) for layer in style_layers.values()]

    noise = torch.Tensor(content_image.size())
    noise = noise.data.normal_(0,0.1).to(device)

    gen_img = (content_image + noise).detach().requires_grad_()
    optimizer = optim.Adam([gen_img], lr=1e-3)
    return gen_img, content_losses_f, style_losses_f, optimizer

def run_iterations(model, requested_content, requested_style, gen_img, content_losses_f, style_losses_f, optimizer, iterations=500, plot=False):
    loop = tqdm(total=iterations, position=0)
    for iteration in range(iterations):
        optimizer.zero_grad()

        gen_layers = model(gen_img)
        loss_1 = 0
        loss_2 = 0
        for layer_name, loss_func in zip(gen_layers, content_losses_f):
            if layer_name in requested_content:
                val = loss_func(gen_layers[layer_name])
                loss_1 += val

        for layer_name, loss_func in zip(gen_layers, style_losses_f):
            if layer_name in requested_style:
                val = loss_func(gen_layers[layer_name])
                w_l = 1/5
                loss_2 += val*w_l

        total_loss = loss(loss_1, loss_2)
        total_loss.backward()

        loop.set_description('cont_loss {:.3f} style_loss {:.3f} loss:{:.4f}'.format(loss_1, loss_2, total_loss.item()))
        loop.update(1)

        optimizer.step()

        if plot and iteration%500==0 and iteration>0:
            plot_img(gen_img)
    return gen_img
