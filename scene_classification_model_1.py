import pandas as pd 
import os 
import random
import time 
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms, models
from preprocessing import Preprocessing
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torchsummary import summary

# 30% test 70% train total 24335 images (test=7301) (train=17034)
dataset = pd.read_csv('DL-FinalProject/scene_classification/train-scene classification/train.csv')

#80-20 split 
def get_classification_data(dataset):
    train_data = []
    test_data = []

    labels = []
    dataset = dataset.groupby('label')
    print(len(dataset))
    for label, image in dataset:
        labels.append(label)
        train_sample = random.sample(list(image['image_name']), 5)
        train_tuple_sample = [(label,  os.path.abspath(os.getcwd()) + '/DL-FinalProject/scene_classification/train-scene classification/train/' + img) for img in train_sample]
        train_data.extend(train_tuple_sample)
        test_sample = random.sample(list(set(image['image_name']) - set(train_sample)), 20)
        test_sample = [(label, os.path.abspath(os.getcwd()) + '/DL-FinalProject/scene_classification/train-scene classification/train/' + img) for img in test_sample]
        test_data.extend(test_sample)

    return train_data, test_data

train_data, test_data = get_classification_data(dataset=dataset)


transformation = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.CenterCrop((224,224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomPosterize(bits=4,p=0.2),
    # transforms.RandomAdjustSharpness(sharpness_factor=2),
    # transforms.ColorJitter(saturation=0.1),
    # transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(train_data, transform=transformation)
test_dataset = CustomDataset(test_data, transform=transformation)

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

device = torch.device('mps')

resnet18_model = torch.load('/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/resnet18_model.pth')

# resnet18_model = model_checkpoint['model']
# state_dict = torch.load('/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/resnet_miniImageNet.pt', map_location=device)
for param in resnet18_model.parameters():
    param.requires_grad = False
input_features = resnet18_model.fc.in_features
output_classes = 6
# resnet18_model.fc = nn.Sequential(
#     nn.Linear(resnet18_model.fc.in_features, 512),
#     nn.ReLU(),
#     nn.Dropout(p=0.4),
#     nn.Linear(512, out_features=output_classes)
# )
resnet18_model.fc = nn.Linear(input_features, output_classes)
print(summary(resnet18_model))
# print(resnet18_model.fc)

# Move model to MPS device
resnet18_model = resnet18_model.to(device)

loss_function = nn.CrossEntropyLoss()

# Update optimizer to work with MPS tensors
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)

def train_nn(model, train_dataloader, test_dataloader, loss_function, optimizer, n_epochs): 
    start_time = time.time() 
    for epoch in range(n_epochs):
        print(f'Epoch number: {epoch + 1}')
        model.train()
        running_loss = 0 
        running_correct = 0 
        total = 0 

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

        # Calculate average loss and accuracy
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100.0 * running_correct / total

        print(f'Training dataset got {running_correct} out of {total} images correctly with Accuracy: {round(epoch_accuracy, 3)} | Epoch loss: {round(epoch_loss, 3)}')
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

train_nn(resnet18_model, train_dataloader, test_dataloader, loss_function, optimizer, 100)

