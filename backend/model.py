import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from PIL import Image


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()

        self.loss_function = nn.MSELoss()
        self.height = height
        self.width = width

        # Encoder layer
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder Layer
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x[:, :, :self.height, :self.width]
        return x

    def pad_img(self, image):
        """
        Adds padding to an image until it is the correct height and width

        Arguments:
            - image -- Image to be padded

        Returns:
            padded_image -- Padded image

        """
        curr_width, curr_height = image.size
        x_offset = (self.width - curr_width) // 2
        y_offset = (self.height - curr_height) // 2
        padded_image = Image.new('L', (self.width, self.height), 0)
        padded_image.paste(image, (x_offset, y_offset))
        return np.array(padded_image)

    def preprocess(self, image_directories):
        """
        Processes the images so that the model can take them in

        Arguments:
            - image_directories -- Image file paths

        Returns:
            - x_images -- Noisy input of shape (batch_size, channels, height, width)
            - y_images -- Truth images of shape (batch_size, channels, height, width)

        """
        x_images = []
        y_images = []
        for image_directory in image_directories:
            image = Image.open(image_directory)
            image = np.array(image)
            if image.shape != (512, 512):
                print(f"{image_directory} has the shape of {image.shape}")
            else:
                image = image / 255
                x_image = add_rician_noise(image, 1)
                image = np.expand_dims(image, axis=0)
                x_image = np.expand_dims(x_image, axis=0)
                x_images.append(x_image)
                y_images.append(image)
        return (x_images, y_images)

    def train_model(self, x, optimiser):
        """
        Trains the model

        Arguments:
            - x -- Image file paths

        Returns:
            self {AutoEncoder} -- Trained model
        """
        self.train()
        EPOCHS = 10
        BATCH_SIZE = round(len(x) / 10)
        losses = []
        epoch_losses = []
        for epoch in range(EPOCHS):
            print(f"Epoch: {epoch + 1}")
            # Shuffle the data
            shuffled_indices = np.random.permutation(len(x))
            shuffled_x = x[shuffled_indices]
            batches = np.array_split(shuffled_x, BATCH_SIZE)

            epoch_loss = 0
            for batch in batches:
                x_batch, y = self.preprocess(batch)
                x_batch = torch.from_numpy(np.array(x_batch))
                y = torch.from_numpy(np.array(y))
                x_batch = x_batch.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                optimiser.zero_grad()
                y_hat = self.forward(x_batch)
                loss = self.loss_function(y_hat, y)
                loss.backward()
                optimiser.step()
                losses.append(loss.item())
                epoch_loss += loss.item()

            print(f"Epoch Training loss: {epoch_loss / len(batches)}")
            epoch_losses.append(epoch_loss / len(batches))

        plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
        plt.xlabel('Total Images')
        plt.ylabel('Loss')
        plt.title('Training Loss over Total Images')
        plt.legend()
        plt.show()

        plt.plot(range(1, EPOCHS + 1), epoch_losses, label="Training Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()

        return self

    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            BATCH_SIZE = round(len(x) / 10)
            batches = np.array_split(x, BATCH_SIZE)
            avg_loss = 0
            for batch in batches:
                x_batch, y = self.preprocess(batch)
                x_batch = torch.from_numpy(np.array(x_batch))
                y = torch.from_numpy(np.array(y))
                x_batch = x_batch.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                y_hat = self.forward(x_batch)
                loss = self.loss_function(y_hat, y)
                avg_loss += loss.item()
            print(f"Evaluation loss: {avg_loss / len(batches)}")


def getImageDirectories(is_train):
    # Images are from here: https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images?select=GAN-Traning+Images
    PATH = "/content/drive/MyDrive/Horizons/images/resized_images/"
    if is_train:
        PATH = PATH + "training/"
    else:
        PATH = PATH + "evaluation/"
    image_names = os.listdir(PATH)
    return np.array(list(map(lambda image: PATH + image, image_names)))


def add_rician_noise(image, snr):
    """
    Add Rician noise to an image.

    Parameters:
    - image: Input image (2D numpy array).
    - snr: Signal-to-Noise Ratio (SNR) of the image.

    Returns:
    - noisy_image: Image with added Rician noise.
    """
    # Calculate the standard deviation of the noise
    sigma = np.sqrt(np.mean(image**2) / snr)

    # Generate Gaussian noise for real and imaginary parts
    real_noise = np.random.normal(0, sigma, image.shape)
    imag_noise = np.random.normal(0, sigma, image.shape)

    # Add noise to the real and imaginary parts of the image
    noisy_real = image + real_noise
    noisy_imag = imag_noise

    # Combine real and imaginary parts to get the magnitude image with Rician noise
    noisy_image = np.sqrt(noisy_real**2 + noisy_imag**2)

    return noisy_image
