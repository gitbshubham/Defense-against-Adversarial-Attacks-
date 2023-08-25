<div id="top"></div>

<br/>
<div align="center">

<h2 align="center">Defense against Adversarial Attacks</h2>

  <p align="center">
    Deep Neural Networks are notoriously known for being very overconfident in their predictions. Szegedy et. al. (https://arxiv.org/abs/1312.6199) discovered that Deep Neural Networks can be fooled into making wrong predictions by adding small perturbations to the original image. In our project, we aim to make targeted classifier models more robust to adversarial attacks. We train an autoencoder, and use this model to effectively counter the adversarial perturbations that have been added to the input image. We also explore defense by generating images that are as close as possible to the input adversarial image. We implement our own attacks and train our own baselines to ensure uniform comparison.
    <br />
    <br />
    <a href="https://github.com/malayp717/dlcv_project">View Demo</a>
    ·
    <a href="https://github.com/malayp717/dlcv_project/issues">Report Bug</a>
    ·
    <a href="https://github.com/malayp717/dlcv_project/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#problem-definition">Problem Definition</a>     </li>
    <li> <a href="#code-execution">Code Execution</a> </li>
    <li><a href="#datasets-used">Datasets Used</a></li>
    <li>
      <a href="#proposed-approach">Proposed Approach</a> </li>
        <li><a href="#experiments">Experiments</a></li>
        <li><a href="#future-directions">Future Direction</a></li>
      </ol>
</details>

<!-- PROBLEM DEFINITION -->
## Problem Definition
In the field of machine learning, building models that are capable of withstanding adversarial attacks has become a crucial area of research. An adversarial attack is an attempt to exploit vulnerabilities in a machine learning model by introducing malicious inputs or perturbations that are designed to deceive the model into making incorrect predictions.

Adversarial attacks are imperceptible changes to inputs which cause neural networks to fail at the intended task, for example classification of images. There has been a good amount of progress in research pertaining to defending against these attacks. The defenses aim to either detect whether an input is adversarial, or aim to modify the input so that it is no longer adversarial in nature. Adversarial attackers on image classifiers fool neural networks by perturbing inputs in the direction of the gradient of the target model. As a result, the perturbed image is a data-point that maps to a different distribution than that of the original image dataset.

We explore two kinds of adversarial attacks namely **Fast Gradient Sign Method (FGSM)** and **Projected Gradient Descent (PGD)** on the MNIST and CIFAR-10 dataset and defend these using variational auto-encoders.

<img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/example.png" />

<!-- CODE EXECUTION -->
## Code Execution
Download the pre-trained models from here (https://drive.google.com/file/d/1iLZd5cCLaI_-KDeW25TwVOocCwgjj0-0/view?usp=share_link)  
For analysis of the adversarial attacks, execute the following `.ipynb` files - 
 - **FGSM Attack** :
      - `fgsm_cifar10.ipynb` for analysis on the CIFAR-10 dataset
      - `fgsm_mnist.ipynb` for analysis on the MNIST dataset
 - **PGD Attack** :
      - `pgd_cifar10.ipynb` for analysis on the CIFAR-10 dataset
      - `pgd_mnist.ipynb` for analysis on the MNIST dataset
      
 For training the autoencoder, execute the following `.ipynb` files - 
 - `ae_cifar10_train.ipynb` for training on the CIFAR-10 dataset
 - `ae_mnist_train.ipynb` for training on the MNIST dataset
 
 For getting the inference result, execute the following `.ipynb` files - 
 - `ae_cifar10_inference.ipynb` for training on the CIFAR-10 dataset
 - `ae_mnist_inference.ipynb` for training on the MNIST dataset

Note : Since we have trained most of our models online on platforms like Google Colab and Kaggle, in some of the `.ipynb` files, the file path maybe a bit incorrect.


<!-- DATASETS USED -->
## Datasets Used
We have used the publicly available **MNIST** database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples (https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and the **CIFAR-10** dataset consisting of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class (https://www.kaggle.com/c/cifar-10)

<img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/datasets.png" />


<!-- PROPOSED APPROACH -->
## Proposed Approach
### Adversarial Attacks
 1. **Fast Gradient Sign Method (FGSM)** : The FGSM exploits the gradients of a neural network to build an adversarial image. Essentially, FGSM computes the gradients of a loss function (e.g., mean-squared error or categorical cross-entropy) with respect to the input image and then uses the sign of the gradients to create a new image (i.e., the adversarial image) that maximizes the loss. <br/><img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/fgsm_eq.png" width="35%"/> <br/> <img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/fgsm.png" />
 2. **Projected Gradient Descent (PGD)** : Akin to i-FGSM i.e “many mini FGSM steps”. A slight difference lies in the optimization algorithm used in this approach, compared to FGSM. FGSM uses normal gradient descent steps. PGD in contrast, runs projected (normalized) steepest descent under the l∞ norm. <img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/pgd.png"/>

### Proposed Defense Architecture 
<img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/defense_arc.png"/>
<img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/performance.png"/>
<img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/table.png"/>

### Inference Time
<img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/inference_arc.png"/>


<!-- EXPERIMENTS  -->
## Experiments
Our initial approach was a **CWGAN + GP** model. A GAN model which learns the adversarial function and applies the inverse of the perturbation applied so that we can get an image close to the real image. The loss function used was **Wasserstein Distance** in order to avoid the mode collapse of the GAN. We applied transfer learning to already existing architectures for classification (AlexNet, ResNet50, VGG-16, etc) and use one round of defensive distillation in the last layer. 
 - No need to train on adversarial images, just need to train on the original dataset, and calculate the MSE loss at inference time
 - More robust, since the inherent assumption while doing adversarial attacks is that the perturbation is small. Therefore, adversarial training is not needed
 - At inference, we just need to find the label corresponding to the minimum loss with the adversarial image, so it can defend against multiple attacks at once
 <img src="https://github.com/malayp717/dlcv_project/blob/master/pictures/prev_arc.png"/>
 
**Why did this not work?**
- GANs are notoriously hard to train, because of alternating optimization (modal collapse, can diverge, etc), and need a very large number of epochs to learn the original dataset distribution.
 - FID score is used to check GANs accuracy, the closer to 0, the better the model. After training our model on MNIST dataset for around 70 epochs (30 mins per epoch), the FID score was upwards from 100. In some papers, the GANs are trained for as long as 300-400 epochs.
 <img src="https://github.com/malayp717/dlcv_project/blob/master/plots/cwgan_gp_loss_mnist.png"/>


<!-- FUTURE DIRECTIONS  -->
## Future Directions

- Provided enough computation power, we can train our model for longer duration to get lower FID scores.
- Instead of using MSE loss at inference time, we can train Siamese network on the (real, adversarial) image combination, and compute the similarity scores at inference to get the true label corresponding to the adversarial image.
- Since GANs are very hard to train, we can use other alternatives of generative (conditional) models like Variational Autoencoders (VAEs), Diffusion Models. Recent advances in Diffusion Models are known to achieve low FID scores (lower than 10).
