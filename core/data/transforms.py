import torch
import torchvision.transforms as T


class RandomCrop:
    """
    A wrapper for torchivision.transforms.RandomCrop
    """

    def __init__(self, size):
        self.size = size
        self.transform = T.RandomCrop(size)

    def __call__(self, img):
        return self.transform(img)

class ToTensor:
    """
    Converts a PIL image or numpy.ndarray to atorch.Tensor
    """

    def __init__(self):
        self.transform = T.ToTensor()

    def __call__(self, img):
        return self.transform(img)


class Normalize:
    """
    Normalize a PIL image or numpy.ndarray
    """
    def __init__(self, mean, std):
        self.transform = T.Normalize(mean,std)

    def __call__(self, tensor):
        return self.transform(tensor)

class AddGaussianNoise:
    """
    Adds gaussian noise to an image.
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * (self.sigma/255.0)
        return torch.clamp(tensor + noise, 0.0, 1.0)