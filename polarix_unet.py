import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as alb
from torch.utils.data import DataLoader

def get_augs():
    return alb.Compose([
      alb.Resize(IMAGE_SIZE_X, IMAGE_SIZE_Y)
      ], is_check_shapes=False)

def get_val_augs():
    return alb.Compose([
      alb.Resize(IMAGE_SIZE_X, IMAGE_SIZE_Y),
      ], is_check_shapes=False)

class SegmentationDataset(Dataset):

    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = row.train_X
        mask = row.train_Y


        if self.augmentations:
            data = self.augmentations(image = image) #, mask = mask)
            image = data['image']
            data = self.augmentations(image = mask)
            mask = data['image']

        image = np.expand_dims(image,axis=0)
        mask = np.expand_dims(mask,axis=0)

        return image, mask

def train_function(data_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for inputs, targets in tqdm(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(inputs, outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_function(data_loader, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(inputs, outputs, targets)

            total_loss += loss.item()

    return total_loss / len(data_loader)


class LossFunction:

    def __init__(self, projection_loss_factor=1e-4, centroid_loss_factor=1e-4):
        self.projection_loss_factor = projection_loss_factor
        self.centroid_loss_factor = centroid_loss_factor
        self.pixel_loss_fn = nn.MSELoss()  # Initialize the pixel loss function once

    def __call__(self, inputs, outputs, targets):
        # Calculate pixel loss
        pixel_loss = self.pixel_loss_fn(outputs, targets)

        # Calculate projection loss
        out_projection = torch.sum(outputs, axis=2)
        target_projection = torch.sum(targets, axis=2)
        projection_loss = self.pixel_loss_fn(out_projection, target_projection)

        # Set up pixel values for centroid calculation
        pixel_values = torch.arange(inputs.shape[2]).unsqueeze(1).repeat(1, inputs.shape[3]).to(inputs.device)

        # Mean value calculation
        mean_value = torch.mean(inputs, dim=2)

        # Weighted average (centroid) calculations
        mean_px_out = torch.sum(pixel_values * outputs, dim=2) / torch.sum(outputs, dim=2)
        mean_px_out[mean_value < 0.005] = 0.0

        mean_px_input = torch.sum(pixel_values * inputs, dim=2) / torch.sum(inputs, dim=2)
        mean_px_input[mean_value < 0.005] = 0.0

        mean_diff = mean_px_out - mean_px_input
        mean_diff = torch.clamp(mean_diff, min=0)

        centroid_loss = torch.mean(mean_diff)

        # Calculate total loss using the factors
        total_loss = (
                pixel_loss
                + self.projection_loss_factor * projection_loss
                + self.centroid_loss_factor * centroid_loss
        )

        return total_loss



train_data = pd.read_pickle("data/train_data_2023_11-08_new.pkl")

train_df, val_df = train_test_split(train_data, test_size = 0.1)

train_X, train_Y = train_df['train_X'], train_df['train_Y']
val_X, val_Y = val_df['train_X'], val_df['train_Y']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE_X, IMAGE_SIZE_Y = 140, 200
BATCH_SIZE = 32

trainset = SegmentationDataset(train_df, get_augs())
valset = SegmentationDataset(val_df, get_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(valset)}")

trainloader = DataLoader(trainset, batch_size= BATCH_SIZE, shuffle = False)
valloader = DataLoader(valset, batch_size= BATCH_SIZE)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        x0 = F.relu(self.conv0(x))

        x1 = F.relu(self.conv1(x0))
        x1 = F.relu(self.conv2(x1))
        x2 = self.pool1(x1)

        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        x3 = self.pool2(x2)

        # Bottleneck
        x3 = F.relu(self.conv5(x3))
        x3 = F.relu(self.conv6(x3))

        # Decoder
        x4 = self.upconv1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = F.relu(self.conv7(x4))
        x4 = F.relu(self.conv8(x4))

        x5 = self.upconv2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = F.relu(self.conv9(x5))
        x5 = F.relu(self.conv10(x5))
        x5 = self.conv11(x5)
        return x5


# Instantiate the model
model = UNet()

# Print the model architecture
print(model)

# model = UNet()

Train = True

if Train:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # criterion = nn.MSELoss()
    criterion = LossFunction()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    # Training loop
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = np.Inf

    train_L, val_L = [], []
    for i in range(EPOCHS):

        train_loss = train_function(trainloader, model, criterion, optimizer)
        train_L.append(train_loss)
        val_loss = eval_function(valloader, model)
        val_L.append(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f'saved_models/best_model{timestamp}.pt')
            best_val_loss = val_loss

        print(f"Epoch: {i + 1}   Train loss: {train_loss}  Val_loss: {val_loss}")

    plt.plot(train_L, label="train loss")
    plt.plot(val_L, label="test loss")
    plt.legend()

else:

    model = UNet()
    state_dict = torch.load("saved_models/best_model_yet.pt")
    model.load_state_dict(state_dict)
    model.to(device)


