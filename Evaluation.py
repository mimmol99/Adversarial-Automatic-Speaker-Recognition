import torch
from tqdm import tqdm
from torch import nn,optim
from matplotlib import pyplot as plt

def calculate_accuracy(scores, labels,threshold = 0.5):
    if scores.dim() == 1 or scores.size(1) == 1:
        # Binary classification case
        # Convert scores to binary predictions (0 or 1)
        predicted = (scores > threshold).int()
    else:
        # Multi-class classification case
        _, predicted = torch.max(scores, 1)

    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def training_phase(model, train_loader, eval_loader, epochs, criterion, optimizer, patience=None,plot=True):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    

    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0

        # Wrap train_loader with tqdm for progress tracking
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            labels = labels.clone().detach()
            scores = model(inputs)

            if isinstance(criterion,nn.BCELoss) or isinstance(criterion,nn.BCEWithLogitsLoss):
                labels = labels.float()

                scores = scores.squeeze(1)

            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(scores, labels)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        model.eval()
        total_eval_loss, total_eval_accuracy = 0, 0

        # Wrap eval_loader with tqdm for progress tracking
        with torch.no_grad():
            for inputs, labels in tqdm(eval_loader, desc=f"Epoch {epoch+1}/{epochs} [Evaluation]"):
                labels = labels.clone().detach()
                scores = model(inputs)

                if isinstance(criterion,nn.BCELoss) or isinstance(criterion,nn.BCEWithLogitsLoss):
                    labels = labels.float()

                    scores = scores.squeeze(1)

                loss = criterion(scores, labels)
                #print("vr",loss)
                total_eval_loss += loss.item()
                total_eval_accuracy += calculate_accuracy(scores, labels)

        avg_eval_loss = total_eval_loss / len(eval_loader)
        avg_eval_accuracy = total_eval_accuracy / len(eval_loader)
        val_losses.append(avg_eval_loss)
        val_accuracies.append(avg_eval_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_eval_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}, Validation Accuracy: {avg_eval_accuracy:.4f}')

        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve == patience:
            print('Early stopping triggered')
            break
            
    if plot == True:
        # Plotting
        plt.figure(figsize=(12, 5))

        # Plotting training and validation accuracies
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
        #plt.axhline(y=avg_test_accuracy, color='r', linestyle='-', label='Test Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xlim(1,epochs)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)

        # Plotting training and validation losses
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        #plt.axhline(y=avg_test_loss, color='r', linestyle='-', label='Test Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim(1,epochs)
        plt.ylim(0, 3)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    return model


def test_phase(model,test_loader,criterion):
    model.eval()
    total_test_loss, total_test_accuracy = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.clone().detach()

            scores = model(inputs)

            if isinstance(criterion,nn.BCELoss) or isinstance(criterion,nn.BCEWithLogitsLoss):
                labels = labels.float()

                scores = scores.squeeze(1)

            loss = criterion(scores, labels)

            total_test_loss += loss.item()
            total_test_accuracy += calculate_accuracy(scores, labels)

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_accuracy = total_test_accuracy / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')

    return model
