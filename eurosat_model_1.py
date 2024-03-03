import os
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
preprocess = Preprocessing()
train_data, test_data = preprocess.euroSat_get_categories(os.getcwd() +'/DL-FinalProject/EuroSAT_RGB')

transformation = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomRotation(11),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ColorJitter(contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(train_data, transform=transformation)
test_dataset = CustomDataset(test_data, transform=transformation)

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)
# for batch_idx, (images, labels) in enumerate(train_dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     # print("Images shape:", images.shape)
#     print("Labels:", labels)

# for batch_idx, (images, labels) in enumerate(test_dataloader):
#     print(f"Batch {batch_idx + 1}:")
#     # print("Images shape:", images.shape)
#     print("Labels:", labels)
# Check if MPS is available
device = torch.device('mps')

resnet18_model = torch.load('/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/resnet18_model.pth')
# resnet18_model = model_checkpoint['model']
# state_dict = torch.load('/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/resnet_miniImageNet.pt', map_location=device)
for param in resnet18_model.parameters():
    param.requires_grad = False
input_features = resnet18_model.fc.in_features
output_classes = 10
# resnet18_model.fc = nn.Sequential(
#     nn.Linear(input_features, 256),
#     nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(256, out_features=output_classes)
# )
resnet18_model.fc = nn.Linear(input_features, output_classes)
print(summary(resnet18_model))
# state_dict['fc.weight'] = state_dict['fc.weight'][:output_classes, :]
# state_dict['fc.bias'] = state_dict['fc.bias'][:output_classes]
# resnet18_model.load_state_dict(state_dict, strict=False)

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
            # print('Prediction:', predicted, 'Correct Answer:', labels)
    epoch_accuracy = 100.0 * predicted_correctly_on_epoch / total
    print(f'Testing dataset got {predicted_correctly_on_epoch} out of {total} images correctly with Accuracy: {round(epoch_accuracy, 3)}')

train_nn(resnet18_model, train_dataloader, test_dataloader, loss_function, optimizer, 100)

