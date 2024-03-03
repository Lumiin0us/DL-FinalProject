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

train_data, valid_data, test_data = preprocess.train_test_valid_split(preprocess.get_images(os.getcwd() + '/DL-FinalProject/mini-ImageNet'))

transformation = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    # transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(train_data, transform=transformation)
valid_dataset = CustomDataset(valid_data, transform=transformation)
test_dataset = CustomDataset(test_data, transform=transformation)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
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

mobileNetV2_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)

# state_dict = torch.load('/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/resnet_miniImageNet.pt', map_location=device)
# for param in mobileNetV2_model.parameters():
#     param.requires_grad = False
input_features = mobileNetV2_model.classifier[-1].in_features
output_classes = 64
# mobileNetV2_model.fc = nn.Sequential(
#     nn.Linear(input_features, 256),
#     nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(256, out_features=output_classes)
# )
# mobileNetV2_model.classifier[-1] = nn.Linear(input_features, output_classes)
mobileNetV2_model.classifier[-1] = nn.Linear(input_features, output_classes)
print(summary(mobileNetV2_model))
# state_dict['fc.weight'] = state_dict['fc.weight'][:output_classes, :]
# state_dict['fc.bias'] = state_dict['fc.bias'][:output_classes]
# mobileNetV2_model.load_state_dict(state_dict, strict=False)

# Move model to MPS device
mobileNetV2_model = mobileNetV2_model.to(device)

loss_function = nn.CrossEntropyLoss()

# Update optimizer to work with MPS tensors
optimizer = optim.SGD(mobileNetV2_model.parameters(), lr=0.001, weight_decay=0.001, momentum=0.87)

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

train_nn(mobileNetV2_model, train_dataloader, test_dataloader, loss_function, optimizer, 5)

torch.save(mobileNetV2_model, '/Users/abdurrehman/Desktop/DL-FinalProject/DL-FinalProject/pretrained_models/mobileNetV2_model.pth')