import torchvision.transforms as trans
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import glob
import random
import os
import pandas as pd
from numpy import iscomplexobj
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision.io import read_image 
from PIL import Image
import torch
import numpy as np
from numpy import cov
from scipy.linalg import sqrtm
from numpy import trace
from numpy.random import random
import os
import cv2


class LandscapeDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, datatype='train'):
        self.transform = transforms_
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, datatype+'A') + '/*'))
        self.files_B = sorted(glob.glob(os.path.join(root, datatype+'B') + '/*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


act1 = random(2048)
# act1 = act1.reshape((2048))
act2 = random(2048)
# act2 = act2.reshape((10,2048))

print('%.3f' % calculate_frechet_distance(act1, cov(act1), act2, cov(act2)))

batch_size_train = 1
transform_ = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor()
])
dataset_landscape_train = LandscapeDataset('/home/ishika/kaushik/art2real/datasets/landscape2photo/',transforms_ = transform_, datatype = 'train')
test_loader = torch.utils.data.DataLoader(dataset = dataset_landscape_train, batch_size=batch_size_train, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.avgpool.register_forward_hook(get_activation('avgpool'))
model.eval()
# model = torch.load('/home/ishika/kaushik/art2real/checkpoints/landscape2photo/100_net_G_A.pth')

# print(model)
l = []
for i, data in enumerate(test_loader,0):
    output = model(data['B'])
    act1 = torch.squeeze(activation['avgpool']).cpu().numpy()
    l.append(act1.reshape(2048))
    # output = model(data['B'])
    # act2 = torch.squeeze(activation['drop']).cpu().numpy()
    # val = calculate_fid(act1, act2)

# Output = []
# for i in range(len(l)): 
#    Output.append(np.average(l[i])) 

# mean_op = np.array(Output)
mean_op = np.mean(l, axis= 0)
print(mean_op)
m = []
directory = '/home/ishika/kaushik/art2real/resultsfru/landscape2photo/test_latest/images/'

fidi = 0
count1 = 0
for filename in os.listdir(directory):

    if filename.endswith('_fake.png'):
        f = os.path.join(directory, filename)
        im = transform_(Image.fromarray(cv2.imread(f))).view(1,3,299,299)
        output = model(im)
        act1 = torch.squeeze(activation['avgpool']).cpu().numpy()
        m.append(act1.reshape(2048))
        # fidi += calculate_frechet_distance(act1, cov(act1), mean_op, cov(mean_op))
        count1+=1

    # output = model(data['B'])
    # act1 = torch.squeeze(activation['drop']).cpu().numpy()
    # l.append(act1)

# Output1 = []
# for i in range(len(m)): 
#    Output1.append(np.average(m[i])) 

# mean_op1 = np.array(Output1)
mean_op1 = np.mean(m, axis= 0)
print(mean_op1)

# print(l[0].shape, m[0].shape)
# print(mean_op1.shape, mean_op.shape)
fidi = calculate_frechet_distance(mean_op, cov(mean_op), mean_op1, cov(mean_op1))


# print(fidi/count1)
print(fidi)
# print('val', val)