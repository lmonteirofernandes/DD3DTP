import os
import random
import socket
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset, Dataset

feature_map_augmentation = True
double_supervision = True

# Dataset and results directories
dataset_dir='./datasets/'
results_dir='./results/'

# Dataset sizes
n_data_train = 8000
n_data_val = 2000

# Hyperparameters
batch_size = 5
learning_rate_scalar_field_predictor = 1e-5
learning_rate_scalar_predictor = 5e-8
num_epochs = 100000
patience=10
encoder_channels=[256, 512, 1024]
conv_dim=128
mlp_dim=256
early_stopping = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("===> Using device:", device)

# Create results directory structure
if os.path.exists(results_dir):
    os.system('rm -rf '+results_dir)
    os.makedirs(results_dir)
else:
    os.makedirs(results_dir)

os.makedirs(results_dir+'predictions', exist_ok=True)
os.makedirs(results_dir+'predictions/correlation_diags', exist_ok=True)
os.makedirs(results_dir+'predictions/train', exist_ok=True)
os.makedirs(results_dir+'predictions/val', exist_ok=True)
os.makedirs(results_dir+'predictions/train/phi', exist_ok=True)
os.makedirs(results_dir+'predictions/val/phi', exist_ok=True)
os.makedirs(results_dir+'predictions/train/tough', exist_ok=True)
os.makedirs(results_dir+'predictions/val/tough', exist_ok=True)
os.makedirs(results_dir+'losses/images', exist_ok=True)
os.makedirs(results_dir+'losses/train', exist_ok=True)
os.makedirs(results_dir+'losses/val', exist_ok=True)
os.makedirs(results_dir+'losses/train/phi', exist_ok=True)
os.makedirs(results_dir+'losses/val/phi', exist_ok=True)
os.makedirs(results_dir+'losses/train/tough', exist_ok=True)
os.makedirs(results_dir+'losses/val/tough', exist_ok=True)
os.makedirs(results_dir+'losses/train/total', exist_ok=True)
os.makedirs(results_dir+'losses/val/total', exist_ok=True)
os.makedirs(results_dir+'models', exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, path="checkpoint.pt"):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_c, out_c, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, x):
        return self.up(x)


class UNet3D_upsample_periodic_conv(nn.Module):
    def __init__(self, encoder_channels=[16, 32, 64]):
        super().__init__()
        self.encoder_channels = encoder_channels

        # Encoder
        self.encoders = nn.ModuleList()
        current_in = 4
        for out_c in encoder_channels:
            self.encoders.append(DoubleConv(current_in, out_c))
            current_in = out_c

        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in encoder_channels])
        self.bottleneck = DoubleConv(encoder_channels[-1], 2 * encoder_channels[-1])

        # Decoder
        self.up_layers = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        reversed_channels = encoder_channels[::-1]

        for i, out_c in enumerate(reversed_channels):
            in_c = 2 * reversed_channels[0] if i == 0 else reversed_channels[i - 1]
            self.up_layers.append(UpConv(in_c, out_c))
            self.decoder_convs.append(DoubleConv(2 * out_c, out_c))

        self.final = nn.Conv3d(reversed_channels[-1], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        # Encoder path
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for up, dec_conv, skip in zip(
            self.up_layers, self.decoder_convs, reversed(skips)
        ):
            x = up(x)
            x = torch.cat([x, skip], 1)
            x = dec_conv(x)

        return self.final(x)


class Rotate3D:
    def __init__(self):
        pass

    def __call__(self, tensors):
        input_tensor, target_tensor = tensors

        axis_rotation = {
            0: [0, 180],
            1: [0, 180],
            2: [0, 90, 180, 270],
        }

        for axis in axis_rotation:
            angle = random.choice(axis_rotation[axis])
            if angle == 0:
                continue

            transform = Rotate3D()
            input_tensor, target_tensor = transform.rotate(
                input_tensor, target_tensor, angle, axis
            )

        return input_tensor, target_tensor

    def __call_tensor__(self, tensor):

        axis_rotation = {
            0: [0, 180],
            1: [0, 180],
            2: [0, 90, 180, 270],
        }

        for axis in axis_rotation:
            angle = random.choice(axis_rotation[axis])
            if angle == 0:
                continue

            transform = Rotate3D()
            tensor = transform.rotate_tensor(
                tensor, angle, axis
            )

        return tensor


    def rotate(self, input_tensor, target_tensor, angle, axis):
        """
        Both tensors have shape (C, D, H, W) and should be rotated identically.
        """

        # Convert tensors to numpy (if necessary)
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.cpu().numpy()
        else:
            input_np = input_tensor

        if isinstance(target_tensor, torch.Tensor):
            target_np = target_tensor.cpu().numpy()
        else:
            target_np = target_tensor

        # Define correct rotation plane
        axes_map = {0: (0, 1), 1: (0, 2), 2: (1, 2)}
        rotation_axes = axes_map[axis]

        # Rotate each channel independently
        rotated_input = np.array(
            [
                scipy.ndimage.rotate(
                    ch, angle, axes=rotation_axes, reshape=False, order=1
                )
                for ch in input_np
            ]
        )

        rotated_target = np.array(
            [
                scipy.ndimage.rotate(
                    ch, angle, axes=rotation_axes, reshape=False, order=1
                )
                for ch in target_np
            ]
        )

        # Convert back to tensors
        return (
            torch.tensor(rotated_input, dtype=torch.float32),
            torch.tensor(rotated_target, dtype=torch.float32),
        )

    def rotate_tensor(self, input_tensor, angle, axis):
        """
        Both tensors have shape (C, D, H, W) and should be rotated identically.
        """

        # Define correct rotation plane
        axes_map = {0: (2, 3), 1: (2, 4), 2: (3, 4)}

        rotation_axes = axes_map[axis]

        # Rotate each channel independently
        rotated_input = torch.rot90(input_tensor,k=angle//90, dims=rotation_axes)

        # Convert back to tensors
        return rotated_input


class Translate3D:
    def __init__(self, max_shift=31):
        """
        max_shift: Maximum translation value in each direction (x, y, z)
        """
        self.max_shift = max_shift

    def __call__(self, tensors):
        input_tensor, target_tensor = tensors

        dx = random.randint(0, self.max_shift)
        dy = random.randint(0, self.max_shift)
        dz = random.randint(0, self.max_shift)

        input_tensor = self.translate(input_tensor, dx, dy, dz)
        target_tensor = self.translate(target_tensor, dx, dy, dz)

        return input_tensor, target_tensor

    def __call_tensor__(self, tensor):

        dx = random.randint(0, self.max_shift)
        dy = random.randint(0, self.max_shift)
        dz = random.randint(0, self.max_shift)

        tensor = self.translate_tensor(tensor, dx, dy, dz)

        return tensor

    def translate(self, tensor, dx, dy, dz):
        """
        Apply periodic translation to a 3D tensor.
        tensor: Tensor with shape (C, D, H, W)
        dx, dy, dz: Translation values in each direction
        """
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.cpu().numpy()
        else:
            tensor_np = tensor

        # Apply periodic translation using NumPy roll
        translated_tensor = np.roll(tensor_np, shift=dx, axis=-1)  # Shift along W
        translated_tensor = np.roll(
            translated_tensor, shift=dy, axis=-2
        )  # Shift along H
        translated_tensor = np.roll(
            translated_tensor, shift=dz, axis=-3
        )  # Shift along D

        return torch.tensor(translated_tensor, dtype=torch.float32)

    def translate_tensor(self, tensor, dx, dy, dz):
        """
        Apply periodic translation to a 3D tensor.
        tensor: Tensor with shape (C, D, H, W)
        dx, dy, dz: Translation values in each direction
        """

        # Apply periodic translation using NumPy roll
        translated_tensor = torch.roll(tensor, shifts=dx, dims=-1)  # Shift along W
        translated_tensor = torch.roll(
            translated_tensor, shifts=dy, dims=-2
        )  # Shift along H
        translated_tensor = torch.roll(
            translated_tensor, shifts=dz, dims=-3
        )  # Shift along D

        return translated_tensor

# Network 1: Predicts a 3D scalar field from a 3D image with 4 color channels  ## USE UNET INSTEAD OF THAT
class ScalarFieldPredictor(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(ScalarFieldPredictor, self).__init__()
        # Initial convolution to reduce spatial dimensions
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks without downsampling
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)

        # Final convolution to produce a 3D scalar field of the same size as input
        self.final_conv = nn.Conv3d(512, out_channels, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_conv(x)
        return x

# Network 2: Predicts a single scalar from the 3D scalar field
class ScalarPredictor(nn.Module):
    def __init__(self, conv_dim,mlp_dim, in_channels=1):
        super(ScalarPredictor, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1, padding_mode='circular', bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Dynamically calculate the input size for fc1
        self.fc_input_size = self._get_fc_input_size(in_channels)

        self.fc1 = nn.Linear(self.fc_input_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, 1)

    def _get_fc_input_size(self, in_channels):
        # Create a dummy input to calculate the size after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 32, 32, 32)  # Output size of ScalarFieldPredictor is 32x32x32
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.relu(dummy_output)
            dummy_output = self.maxpool(dummy_output)
            return int(torch.numel(dummy_output))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Combined model and training logic
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.scalar_field_predictor = UNet3D_upsample_periodic_conv(encoder_channels=[32, 64, 128])
        self.scalar_predictor = ScalarPredictor()

    def forward(self, x):
        scalar_field = self.scalar_field_predictor(x)
        scalar_value = self.scalar_predictor(scalar_field)
        return scalar_field, scalar_value


# Loss function
def combined_loss(scalar_field_pred, scalar_value_pred, scalar_field_target, scalar_value_target):
    # Binary cross-entropy for the scalar field prediction
    if double_supervision:
        factor_phi=1e0
    else:
        factor_phi=0
    mse_phi_loss = factor_phi*F.mse_loss(scalar_field_pred, scalar_field_target)
    # Mean squared error for the scalar value prediction
    factor_tough=5e-1
    mse_tough_loss = factor_tough*F.mse_loss(scalar_value_pred, scalar_value_target)
    # Combined loss
    total_loss = mse_phi_loss + mse_tough_loss
    return total_loss, mse_phi_loss, mse_tough_loss


# Function to plot scalar field predictions
def plot_scalar_field(scalar_field_pred, scalar_field_target, epoch, batch_idx, name, dir):
    # Take the first sample in the batch
    scalar_field_pred = scalar_field_pred[0, 0].detach().cpu().numpy()  # Shape: (32, 32, 32)
    scalar_field_target = scalar_field_target[0, 0].detach().cpu().numpy()  # Shape: (32, 32, 32)

    # Plot a slice of the 3D scalar field (e.g., z=16)
    slice_idx = 16
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(scalar_field_pred[:, :, slice_idx], cmap='viridis')
    plt.title(f"Predicted Scalar Field (Epoch {epoch}, Batch {batch_idx})")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(scalar_field_target[:, :, slice_idx], cmap='viridis')
    plt.title(f"Target Scalar Field (Epoch {epoch}, Batch {batch_idx})")
    plt.colorbar()

    # Save the figure
    plt.savefig(os.path.join(dir, f"scalar_field_epoch_{name}_{epoch}_batch_{batch_idx}.png"))

    plt.show()
    plt.close()

# Function to plot correlation diagram
def plot_correlation(true_values, pred_values, epoch, name, dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, pred_values, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--')
    plt.xlabel("True Scalar Values")
    plt.ylabel("Predicted Scalar Values")
    plt.title(f"Correlation Diagram (Epoch {epoch})")
    # Save the figure
    plt.savefig(os.path.join(dir, f"correlation_epoch_{name}_{epoch}.png"))
    plt.show()
    plt.close()
    
    # Save the correlation data
    correlation_data = np.column_stack((true_values, pred_values))
    np.savetxt(os.path.join(results_dir+'predictions/correlation_diags/', f"correlation_epoch_{name}_{epoch}.txt"), correlation_data, delimiter=',', header='True Scalar Values, Predicted Scalar Values', comments='')


# Training loop
def train_and_validate(model_scalar_field_predictor, model_scalar_predictor, dataloader_train, dataloader_val, optimizer_scalar_field_predictor, optimizer_scalar_predictor, device, num_epochs):

    if early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=False)
    total_loss_train_list = []
    total_mse_phi_loss_train_list = []
    total_mse_tough_loss_train_list = []
    total_loss_val_list = []
    total_mse_phi_loss_val_list = []
    total_mse_tough_loss_val_list = []
    for epoch in range(num_epochs):
        model_scalar_field_predictor.train()
        model_scalar_predictor.train()
        total_loss_train = 0.0
        total_mse_phi_loss_train = 0.0
        total_mse_tough_loss_train = 0.0
        all_true_scalars_train = []
        all_pred_scalars_train = []

        for batch_idx, (images, scalar_field_target, scalar_value_target) in enumerate(dataloader_train):
            images = images.to(device)
            scalar_field_target = scalar_field_target.to(device)
            scalar_value_target = scalar_value_target.to(device)

            # Zero gradients
            optimizer_scalar_field_predictor.zero_grad()
            optimizer_scalar_predictor.zero_grad()

            # Forward pass
            scalar_field_pred = model_scalar_field_predictor(images)

            if feature_map_augmentation:
                scalar_value_pred = model_scalar_predictor(rot_layer.__call_tensor__(translate_layer.__call_tensor__(scalar_field_pred)))
            else:
                scalar_field_pred = model_scalar_field_predictor(images)

            # Compute loss
            loss, mse_phi_loss, mse_tough_loss = combined_loss(scalar_field_pred, scalar_value_pred, scalar_field_target, scalar_value_target)

            # Backward pass and optimization
            loss.backward()
            optimizer_scalar_field_predictor.step()
            optimizer_scalar_predictor.step()

            # Accumulate losses
            total_loss_train += loss.item()
            total_mse_phi_loss_train += mse_phi_loss.item()
            total_mse_tough_loss_train += mse_tough_loss.item()

            # Store true and predicted scalar values for correlation plot
            all_true_scalars_train.extend(scalar_value_target.cpu().detach().numpy())
            all_pred_scalars_train.extend(scalar_value_pred.cpu().detach().numpy())

        # Average losses
        avg_loss_train = total_loss_train / len(dataloader_train)
        avg_mse_phi_loss_train = total_mse_phi_loss_train / len(dataloader_train)
        avg_mse_tough_loss_train = total_mse_tough_loss_train / len(dataloader_train)

        # Store losses for plotting
        total_loss_train_list.append(avg_loss_train)
        total_mse_phi_loss_train_list.append(avg_mse_phi_loss_train)
        total_mse_tough_loss_train_list.append(avg_mse_tough_loss_train)

        # Save losses to text files
        with open(os.path.join(results_dir, 'losses/train/total/avg_loss_train.txt'), 'a') as f:
            f.write(f"{avg_loss_train:.5f}\n")
        with open(os.path.join(results_dir, 'losses/train/phi/avg_mse_phi_loss_train.txt'), 'a') as f:
            f.write(f"{avg_mse_phi_loss_train:.5f}\n")
        with open(os.path.join(results_dir, 'losses/train/tough/avg_mse_tough_loss_train.txt'), 'a') as f:
            f.write(f"{avg_mse_tough_loss_train:.5f}\n")

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss_train:.4f}, Train MSE PHI Loss: {avg_mse_phi_loss_train:.4f}, Train MSE TOUGH Loss: {avg_mse_tough_loss_train:.4f}")

        # Plot scalar field predictions
        plot_scalar_field(scalar_field_pred, scalar_field_target, epoch, batch_idx, 'train', results_dir+'predictions/train/phi/')

        # Plot correlation diagram
        plot_correlation(np.array(all_true_scalars_train), np.array(all_pred_scalars_train), epoch, 'train', results_dir+'predictions/train/tough/')

        model_scalar_field_predictor.eval()
        model_scalar_predictor.eval()
        total_loss_val = 0.0
        total_mse_phi_loss_val = 0.0
        total_mse_tough_loss_val = 0.0
        all_true_scalars_val = []
        all_pred_scalars_val = []

        with torch.no_grad():
            for batch_idx, (images, scalar_field_target, scalar_value_target) in enumerate(dataloader_val):
                images = images.to(device)
                scalar_field_target = scalar_field_target.to(device)
                scalar_value_target = scalar_value_target.to(device)

                # Forward pass
                # scalar_field_pred, scalar_value_pred = model(images)

                scalar_field_pred = model_scalar_field_predictor(images)
                # scalar_value_pred = model_scalar_predictor(scalar_field_pred)
                scalar_value_pred = model_scalar_predictor(rot_layer.__call_tensor__(translate_layer.__call_tensor__(scalar_field_pred)))

                # Compute loss
                loss, mse_phi_loss, mse_tough_loss = combined_loss(scalar_field_pred, scalar_value_pred, scalar_field_target, scalar_value_target)

                # Accumulate losses
                total_loss_val += loss.item()
                total_mse_phi_loss_val += mse_phi_loss.item()
                total_mse_tough_loss_val += mse_tough_loss.item()

                # Store true and predicted scalar values for correlation plot
                all_true_scalars_val.extend(scalar_value_target.cpu().detach().numpy())
                all_pred_scalars_val.extend(scalar_value_pred.cpu().detach().numpy())

            # Average losses
            avg_loss_val = total_loss_val / len(dataloader_val)
            avg_mse_phi_loss_val = total_mse_phi_loss_val / len(dataloader_val)
            avg_mse_tough_loss_val = total_mse_tough_loss_val / len(dataloader_val)

            # Save losses to text files
            with open(os.path.join(results_dir, 'losses/val/total/avg_loss_val.txt'), 'a') as f:
                f.write(f"{avg_loss_val:.5f}\n")
            with open(os.path.join(results_dir, 'losses/val/phi/avg_mse_phi_loss_val.txt'), 'a') as f:
                f.write(f"{avg_mse_phi_loss_val:.5f}\n")
            with open(os.path.join(results_dir, 'losses/val/tough/avg_mse_tough_loss_val.txt'), 'a') as f:
                f.write(f"{avg_mse_tough_loss_val:.5f}\n")

            # Store losses for plotting
            total_loss_val_list.append(avg_loss_val)
            total_mse_phi_loss_val_list.append(avg_mse_phi_loss_val)
            total_mse_tough_loss_val_list.append(avg_mse_tough_loss_val)

            print(f"Epoch [{epoch}/{num_epochs}], Val Loss: {avg_loss_val:.4f}, Val MSE PHI Loss: {avg_mse_phi_loss_val:.4f}, Val MSE TOUGH Loss: {avg_mse_tough_loss_val:.4f}")
            
            # Plot scalar field predictions
            plot_scalar_field(scalar_field_pred, scalar_field_target, epoch, batch_idx, 'val',  results_dir+'predictions/val/phi/')

            # Plot correlation diagram
            plot_correlation(np.array(all_true_scalars_val), np.array(all_pred_scalars_val), epoch, 'val',  results_dir+'predictions/val/tough/')

            # Plot all losses
            plt.figure(figsize=(12, 6))
            plt.plot(total_loss_train_list, label='Train Loss (total)')
            plt.plot(total_loss_val_list, label='Validation Loss (total)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            # Save the figure
            plt.savefig(os.path.join(results_dir+'losses/images', f"total_loss_epoch_{epoch}.png"))
            plt.show()
            plt.close()

            # Plot all losses
            plt.figure(figsize=(12, 6))
            plt.plot(total_mse_phi_loss_train_list, label='Train Loss (MSE phi)')
            plt.plot(total_mse_phi_loss_val_list, label='Validation Loss (MSE phi)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            # Save the figure
            plt.savefig(os.path.join(results_dir+'losses/images', f"mse_phi_loss_epoch_{epoch}.png"))
            plt.show()
            plt.close()

            # Plot all losses
            plt.figure(figsize=(12, 6))
            plt.plot(total_mse_tough_loss_train_list, label='Train Loss (MSE tough)')
            plt.plot(total_mse_tough_loss_val_list, label='Validation Loss (MSE tough)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            # Save the figure
            plt.savefig(os.path.join(results_dir+'losses/images', f"mse_tough_loss_epoch_{epoch}.png"))
            plt.show()
            plt.close()

        # Save the model checkpoint
        torch.save(model_scalar_field_predictor.state_dict(), os.path.join(results_dir+'models/', f"scalar_field_predictor_epoch_{epoch}.pth"))
        torch.save(model_scalar_predictor.state_dict(), os.path.join(results_dir+'models/', f"scalar_predictor_epoch_{epoch}.pth"))

        # Early stopping check
        if early_stopping == True:
            early_stopping(avg_loss_val)
            if early_stopping.early_stop:
                print("Early stopping")
                break


# Custom Dataset
class customDataset(Dataset):
    def __init__(
        self,
        data,
        scalar_field,
        scalar_value,
        transform=None,
    ):
        self.data = data
        self.scalar_field = scalar_field
        self.scalar_value = scalar_value
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_img, target_img, target_scalar = self.data[idx], self.scalar_field[idx] , self.scalar_value[idx]

        if self.transform:
            input_img, target_img = self.transform((input_img, target_img))

        return torch.tensor(input_img), torch.tensor(target_img), torch.tensor(target_scalar)

# Load dataset
images=np.load(dataset_dir+'quat_10000_32^3.npy')
scalar_field_target=np.load(dataset_dir+'phi_10000_32^3.npy')
scalar_value_target=np.squeeze(np.load(dataset_dir+'tough_10000_32^3.npy'),axis=(2,3,4))

# Split dataset into training and validation sets
images_train=images[0:n_data_train]
scalar_field_target_train=scalar_field_target[0:n_data_train]
scalar_value_target_train=scalar_value_target[0:n_data_train]
images_val=images[n_data_train:n_data_train+n_data_val]
scalar_field_target_val=scalar_field_target[n_data_train:n_data_train+n_data_val]
scalar_value_target_val=scalar_value_target[n_data_train:n_data_train+n_data_val]

# Normalize dataset
scalar_field_target_train=(scalar_field_target_train-scalar_field_target.mean())/(scalar_field_target.std())
scalar_field_target_val=(scalar_field_target_val-scalar_field_target.mean())/(scalar_field_target.std())
scalar_value_target_train=(scalar_value_target_train-scalar_value_target.mean())/(scalar_value_target.std())
scalar_value_target_val=(scalar_value_target_val-scalar_value_target.mean())/(scalar_value_target.std())

# Convert to PyTorch tensors
images_train = torch.Tensor(images_train)
scalar_field_target_train = torch.Tensor(scalar_field_target_train)
scalar_value_target_train = torch.Tensor(scalar_value_target_train)

# Initialize transformations
transform = transforms.Compose([Rotate3D(), Translate3D()])

# Create DataLoaders
dataset_train = customDataset(images_train, scalar_field_target_train, scalar_value_target_train, transform=None)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))
dataset_val = customDataset(images_val, scalar_field_target_val, scalar_value_target_val, transform=None)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))

# Initialize models and optimizers
model_scalar_field_predictor = UNet3D_upsample_periodic_conv(encoder_channels).to(device)
model_scalar_predictor = ScalarPredictor(conv_dim,mlp_dim).to(device)
optimizer_scalar_field_predictor = optim.Adam(model_scalar_field_predictor.parameters(), lr=learning_rate_scalar_field_predictor)
optimizer_scalar_predictor = optim.Adam(model_scalar_predictor.parameters(), lr=learning_rate_scalar_predictor)

# Initialize feature map augmentation layers
rot_layer=Rotate3D()
translate_layer=Translate3D()

# Training and validation loop
train_and_validate(model_scalar_field_predictor, model_scalar_predictor, dataloader_train, dataloader_val, optimizer_scalar_field_predictor, optimizer_scalar_predictor, device, num_epochs)