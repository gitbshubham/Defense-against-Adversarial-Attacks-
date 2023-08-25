from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def to_numpy_array(tensor):
    '''Convert torch.Tensor to np.ndarray'''
    tensor_ = tensor.cpu()
    tensor_ = tensor_.squeeze(0)
    tensor_ = tensor_.detach().numpy()
    return tensor_

def to_numpy_array_cifar10(tensor):
    '''Convert torch.Tensor to np.ndarray'''
    tensor_ = tensor.squeeze()
    unnormalize_transform = Compose([Normalize(mean=[0, 0, 0], std=[1. / 0.5, 1. / 0.5, 1. / 0.5]),
                                    Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])])
    arr_ = unnormalize_transform(tensor_)
    arr_ = arr_.cpu()
    arr = arr_.permute(1, 2, 0).detach().numpy()
    return arr

def perturb(imgs, eps, data_grads, dataset):
    # Collect the element-wise sign of the data gradient
    sign_data_grads = data_grads.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    adv_imgs = imgs + eps * sign_data_grads
    # Adding clipping to maintain [0,1] range
    if dataset == 'mnist':
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
    # Return the perturbed image
    return adv_imgs

def fgsm_attack(model, imgs, labels, eps, dataset):
    
    imgs.required_grad = True
    
    outputs = model(imgs)
    loss = F.nll_loss(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    data_grads = imgs.grad.data
    
    adv_imgs = perturb(imgs, eps, data_grads, dataset)
    outputs = model(adv_imgs)
    new_preds = outputs.argmax(axis=1)
    
    return adv_imgs, new_preds

def pgd_linf(model, imgs, labels, epsilon, alpha, num_iter):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(imgs, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(imgs + delta), labels)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        new_preds = model(imgs + delta).argmax(axis=1)
    return (imgs + delta).detach(), new_preds

def gradient_penalty(disc, real, fake, device='cpu'):
    # print(real.shape)
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    
    # Calculate Critic scores
    mixed_scores = disc(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) **2)
    return gradient_penalty

def gradient_penalty_conditional(disc, labels, real, fake, device='cpu'):
    # print(real.shape)
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    
    # Calculate Critic scores
    mixed_scores = disc(interpolated_images, labels)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) **2)
    return gradient_penalty