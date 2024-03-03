import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from preprocessing import Preprocessing
from torch.utils.data import Dataset, DataLoader
import time 
# Check if MPS is available
device = torch.device('mps')

preprocess = Preprocessing()
train_data, valid_data, test_data = preprocess.train_test_valid_split(preprocess.get_images('DL-FinalProject/mini-ImageNet'))

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, img_path = self.data[idx]

        # Load the image
        img = Image.open(img_path)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Move data to MPS device
        img = img.to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)

        return img, label
    
# Transformation for images before prediction 
transformation = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    # transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(train_data, transform=transformation)
valid_dataset = CustomDataset(valid_data, transform=transformation)
test_dataset = CustomDataset(test_data, transform=transformation)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Setting model to evaluation mode
resnet18_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
input_features = resnet18_model.fc.in_features
output_classes = 64
# for param in resnet18_model.parameters():
#     param.requires_grad = False
# resnet18_model.fc = nn.Sequential(
#     nn.Linear(input_features, 1000),  # Add an intermediate fully connected layer
#     nn.ReLU(),  # Add a ReLU activation function
#     nn.Dropout(0.5),  # Add a dropout layer with a dropout rate of 0.5 (adjust as needed)
#     # nn.Linear(512, 512),  # Add an intermediate fully connected layer
#     # nn.ReLU(),  # Add a ReLU activation function
#     # nn.Dropout(0.5),
#     nn.Linear(1000, output_classes), # The final fully connected layer
#     nn.LogSoftmax(dim=1),
# )
resnet18_model.fc = nn.Linear(input_features, output_classes)

# Move model to MPS device
resnet18_model = resnet18_model.to(device)

loss_function = nn.CrossEntropyLoss()

# Update optimizer to work with MPS tensors
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.001, momentum=0.9)

# optimizer = optim.Adagrad(resnet18_model.parameters(), lr=0.001, weight_decay=0.0001)
# { 
#     Epoch number: 1
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [13:37<00:00,  1.82s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:29<00:00,  2.03it/s]
# Training dataset got 18186 out of 28800 images correctly with Accuracy: 63.146 | Epoch loss: 1.53
# Validation dataset got 3100 out of 3840 images correctly with Accuracy: 80.729 | Epoch loss: 0.781
# Testing dataset got 4620 out of 5760 images correctly with Accuracy: 80.208
# Epoch number: 2
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:38<00:00,  1.69s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:29<00:00,  2.03it/s]
# Training dataset got 23691 out of 28800 images correctly with Accuracy: 82.26 | Epoch loss: 0.696
# Validation dataset got 3215 out of 3840 images correctly with Accuracy: 83.724 | Epoch loss: 0.619
# Testing dataset got 4795 out of 5760 images correctly with Accuracy: 83.247
# Epoch number: 3
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:24<00:00,  1.65s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:33<00:00,  1.77it/s]
# Training dataset got 25214 out of 28800 images correctly with Accuracy: 87.549 | Epoch loss: 0.489
# Validation dataset got 3247 out of 3840 images correctly with Accuracy: 84.557 | Epoch loss: 0.571
# Testing dataset got 4847 out of 5760 images correctly with Accuracy: 84.149
# Epoch number: 4
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [15:12<00:00,  2.03s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:35<00:00,  1.70it/s]
# Training dataset got 26065 out of 28800 images correctly with Accuracy: 90.503 | Epoch loss: 0.373
# Validation dataset got 3258 out of 3840 images correctly with Accuracy: 84.844 | Epoch loss: 0.542
# Testing dataset got 4904 out of 5760 images correctly with Accuracy: 85.139
# Epoch number: 5
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [14:43<00:00,  1.96s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:34<00:00,  1.72it/s]
# Training dataset got 26795 out of 28800 images correctly with Accuracy: 93.038 | Epoch loss: 0.286
# Validation dataset got 3266 out of 3840 images correctly with Accuracy: 85.052 | Epoch loss: 0.53
# Testing dataset got 4890 out of 5760 images correctly with Accuracy: 84.896
# Epoch number: 6
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:58<00:00,  1.73s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:27<00:00,  2.19it/s]
# Training dataset got 27307 out of 28800 images correctly with Accuracy: 94.816 | Epoch loss: 0.223
# Validation dataset got 3270 out of 3840 images correctly with Accuracy: 85.156 | Epoch loss: 0.526
# Testing dataset got 4937 out of 5760 images correctly with Accuracy: 85.712
# Epoch number: 7
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [11:27<00:00,  1.53s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:26<00:00,  2.25it/s]
# Training dataset got 27686 out of 28800 images correctly with Accuracy: 96.132 | Epoch loss: 0.18
# Validation dataset got 3286 out of 3840 images correctly with Accuracy: 85.573 | Epoch loss: 0.519
# Testing dataset got 4921 out of 5760 images correctly with Accuracy: 85.434
# Epoch number: 8
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [11:56<00:00,  1.59s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:27<00:00,  2.20it/s]
# Training dataset got 27926 out of 28800 images correctly with Accuracy: 96.965 | Epoch loss: 0.144
# Validation dataset got 3269 out of 3840 images correctly with Accuracy: 85.13 | Epoch loss: 0.524
# Testing dataset got 4908 out of 5760 images correctly with Accuracy: 85.208
# Epoch number: 9
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [13:43<00:00,  1.83s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:26<00:00,  2.29it/s]
# Training dataset got 28139 out of 28800 images correctly with Accuracy: 97.705 | Epoch loss: 0.116
# Validation dataset got 3286 out of 3840 images correctly with Accuracy: 85.573 | Epoch loss: 0.512
# Testing dataset got 4903 out of 5760 images correctly with Accuracy: 85.122
# Epoch number: 10
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [11:41<00:00,  1.56s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:29<00:00,  2.00it/s]
# Training dataset got 28322 out of 28800 images correctly with Accuracy: 98.34 | Epoch loss: 0.096
# Validation dataset got 3277 out of 3840 images correctly with Accuracy: 85.339 | Epoch loss: 0.526
# Testing dataset got 4929 out of 5760 images correctly with Accuracy: 85.573
# Epoch number: 11
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [11:34<00:00,  1.54s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:27<00:00,  2.20it/s]
# Training dataset got 28414 out of 28800 images correctly with Accuracy: 98.66 | Epoch loss: 0.083
# Validation dataset got 3289 out of 3840 images correctly with Accuracy: 85.651 | Epoch loss: 0.52
# Testing dataset got 4903 out of 5760 images correctly with Accuracy: 85.122
# Epoch number: 12
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [13:15<00:00,  1.77s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:26<00:00,  2.30it/s]
# Training dataset got 28534 out of 28800 images correctly with Accuracy: 99.076 | Epoch loss: 0.068
# Validation dataset got 3285 out of 3840 images correctly with Accuracy: 85.547 | Epoch loss: 0.531
# Testing dataset got 4937 out of 5760 images correctly with Accuracy: 85.712
# Epoch number: 13
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:40<00:00,  1.69s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:27<00:00,  2.15it/s]
# Training dataset got 28574 out of 28800 images correctly with Accuracy: 99.215 | Epoch loss: 0.058
# Validation dataset got 3277 out of 3840 images correctly with Accuracy: 85.339 | Epoch loss: 0.53
# Testing dataset got 4939 out of 5760 images correctly with Accuracy: 85.747
# Epoch number: 14
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:15<00:00,  1.63s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:31<00:00,  1.88it/s]
# Training dataset got 28623 out of 28800 images correctly with Accuracy: 99.385 | Epoch loss: 0.05
# Validation dataset got 3297 out of 3840 images correctly with Accuracy: 85.859 | Epoch loss: 0.525
# Testing dataset got 4917 out of 5760 images correctly with Accuracy: 85.365
# Epoch number: 15
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [11:38<00:00,  1.55s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:31<00:00,  1.93it/s]
# Training dataset got 28671 out of 28800 images correctly with Accuracy: 99.552 | Epoch loss: 0.043
# Validation dataset got 3290 out of 3840 images correctly with Accuracy: 85.677 | Epoch loss: 0.536
# Testing dataset got 4913 out of 5760 images correctly with Accuracy: 85.295
# Finished in 12605.946423768997
# }

# optimizer = optim.RMSprop(resnet18_model.parameters(), lr=0.001, weight_decay=0.0001)
# {
#     Epoch number: 1
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [13:44<00:00,  1.83s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:29<00:00,  2.04it/s]
# Training dataset got 823 out of 28800 images correctly with Accuracy: 2.858 | Epoch loss: 4.107
# Validation dataset got 177 out of 3840 images correctly with Accuracy: 4.609 | Epoch loss: 3.957
# Testing dataset got 248 out of 5760 images correctly with Accuracy: 4.306
# }

# optimizer = optim.Adam(resnet18_model.parameters(), lr=0.001, weight_decay=0.0001)
# {
#     Epoch number: 1
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [19:09<00:00,  2.55s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:33<00:00,  1.79it/s]
# Training dataset got 10685 out of 28800 images correctly with Accuracy: 37.101 | Epoch loss: 2.386
# Validation dataset got 1786 out of 3840 images correctly with Accuracy: 46.51 | Epoch loss: 1.98
# Testing dataset got 2720 out of 5760 images correctly with Accuracy: 47.222
# Epoch number: 2
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [25:27<00:00,  3.40s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:35<00:00,  1.69it/s]
# Training dataset got 15666 out of 28800 images correctly with Accuracy: 54.396 | Epoch loss: 1.693
# Validation dataset got 1824 out of 3840 images correctly with Accuracy: 47.5 | Epoch loss: 2.266
# Testing dataset got 2723 out of 5760 images correctly with Accuracy: 47.274
# }

def train_nn(model, train_dataloader, valid_dataloader, test_dataloader, loss_function, optimizer, n_epochs): 
    start_time = time.time() 
    for epoch in range(n_epochs):
        print(f'Epoch number: {epoch + 1}')
        model.train()
        running_loss = 0 
        running_correct = 0 
        total = 0 
        
        # for data in train_dataloader:
        #     images, labels = data 
        #     total += labels.size(0)

        #     # Move data to MPS device
        #     images, labels = images.to(device), labels.to(device)

        #     optimizer.zero_grad()
        #     outputs = model(images)

        #     _, predicted = torch.max(outputs.data, 1)

        #     loss = loss_function(outputs, labels)
        #     loss.backward()

        #     optimizer.step()
        #     running_loss += loss.item()
        #     running_correct += (labels == predicted).sum().item()
        progress_bar_training = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (images, labels) in progress_bar_training:
            # Move data to MPS device
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

            # Update the progress bar
            # progress_bar.set_postfix({'Epoch Loss': running_loss / (batch_idx + 1), 'Epoch Accuracy': running_correct/(batch_idx + 1)})

        model.eval()
        valid_loss = 0.0
        running_correct_valid = 0
        total_valid = 0
        progress_bar_validation = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
        with torch.no_grad():
            for valid_idx, (images_valid, labels_valid) in progress_bar_validation:
                images_valid, labels_valid = images_valid.to(device), labels_valid.to(device)
                total_valid += labels_valid.size(0)

                outputs_valid = model(images_valid)
                loss_valid = loss_function(outputs_valid, labels_valid)
                valid_loss += loss_valid.item() 

                _, predicted_valid = torch.max(outputs_valid.data, 1)
                running_correct_valid += (predicted_valid == labels_valid).sum().item()

        # Calculate average loss and accuracy
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100.0 * running_correct / total
        epoch_loss_valid = valid_loss / len(valid_dataloader)
        epoch_accuracy_valid = 100.0 * running_correct_valid / total_valid

        print(f'Training dataset got {running_correct} out of {total} images correctly with Accuracy: {round(epoch_accuracy, 3)} | Epoch loss: {round(epoch_loss, 3)}')
        print(f'Validation dataset got {running_correct_valid} out of {total_valid} images correctly with Accuracy: {round(epoch_accuracy_valid, 3)} | Epoch loss: {round(epoch_loss_valid, 3)}')
        evaluate_model(model, test_dataloader)
    end_time = time.time() - start_time
    print(f'Finished in {end_time}')
    return model 

def evaluate_model(model, test_dataloader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0 
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data 
            total += labels.size(0)

            # Move data to MPS device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item() 
    
    epoch_accuracy = 100.0 * predicted_correctly_on_epoch / total
    print(f'Testing dataset got {predicted_correctly_on_epoch} out of {total} images correctly with Accuracy: {round(epoch_accuracy, 3)}')

train_nn(resnet18_model, train_dataloader, valid_dataloader, test_dataloader, loss_function, optimizer, 5)
torch.save(resnet18_model, '/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/pretrained_models/resnet18_model.pth')

# Epoch number: 1
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [14:35<00:00,  1.94s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:34<00:00,  1.74it/s]
# Training dataset got 9354 out of 28800 images correctly with Accuracy: 32.479 | Epoch loss: 3.091
# Validation dataset got 2812 out of 3840 images correctly with Accuracy: 73.229 | Epoch loss: 1.518
# Testing dataset got 4168 out of 5760 images correctly with Accuracy: 72.361
# Epoch number: 2
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [15:03<00:00,  2.01s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:31<00:00,  1.88it/s]
# Training dataset got 19895 out of 28800 images correctly with Accuracy: 69.08 | Epoch loss: 1.273
# Validation dataset got 3121 out of 3840 images correctly with Accuracy: 81.276 | Epoch loss: 0.781
# Testing dataset got 4661 out of 5760 images correctly with Accuracy: 80.92
# Epoch number: 3
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:30<00:00,  1.67s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:29<00:00,  2.01it/s]
# Training dataset got 22500 out of 28800 images correctly with Accuracy: 78.125 | Epoch loss: 0.826
# Validation dataset got 3221 out of 3840 images correctly with Accuracy: 83.88 | Epoch loss: 0.608
# Testing dataset got 4787 out of 5760 images correctly with Accuracy: 83.108
# Epoch number: 4
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [15:09<00:00,  2.02s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:38<00:00,  1.55it/s]
# Training dataset got 23776 out of 28800 images correctly with Accuracy: 82.556 | Epoch loss: 0.641
# Validation dataset got 3242 out of 3840 images correctly with Accuracy: 84.427 | Epoch loss: 0.55
# Testing dataset got 4855 out of 5760 images correctly with Accuracy: 84.288
# Epoch number: 5
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [12:31<00:00,  1.67s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:27<00:00,  2.17it/s]
# Training dataset got 24625 out of 28800 images correctly with Accuracy: 85.503 | Epoch loss: 0.526
# Validation dataset got 3294 out of 3840 images correctly with Accuracy: 85.781 | Epoch loss: 0.511
# Testing dataset got 4918 out of 5760 images correctly with Accuracy: 85.382
# Epoch number: 6
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [13:21<00:00,  1.78s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:32<00:00,  1.86it/s]
# Training dataset got 25294 out of 28800 images correctly with Accuracy: 87.826 | Epoch loss: 0.442
# Validation dataset got 3310 out of 3840 images correctly with Accuracy: 86.198 | Epoch loss: 0.482
# Testing dataset got 4928 out of 5760 images correctly with Accuracy: 85.556
# Epoch number: 7
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 450/450 [13:31<00:00,  1.80s/it]
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:28<00:00,  2.07it/s]
# Training dataset got 25872 out of 28800 images correctly with Accuracy: 89.833 | Epoch loss: 0.369
# Validation dataset got 3306 out of 3840 images correctly with Accuracy: 86.094 | Epoch loss: 0.478
# Testing dataset got 4923 out of 5760 images correctly with Accuracy: 85.469