#Evaluation methods

import torch
from tqdm import tqdm
from torch import nn,optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import librosa


def get_criterion(num_classes):
    if num_classes == 1:
        return nn.BCELoss()
        #return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()


def get_optimizer(model,num_classes,lr = 1e-5):
    if num_classes == 1:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        return optim.Adam(model.parameters(), lr=lr)


def signal_to_spectogram(signal,sr = 16000,n_mels = 80,to_db = True):

    if not isinstance(signal,np.ndarray):
        signal = np.array(signal)

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)

    if to_db:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec

def signals_to_spectogram(signals,sr=16000,n_mels=80,to_db = True):
    mel_specs = [signal_to_spectogram(signal,sr=sr,n_mels=n_mels,to_db=to_db) for signal in signals]
    return mel_specs



def calculate_accuracy(scores, labels,threshold = 0.5):

    if scores.dim() == 1 or scores.size(1) == 1:
        # Binary classification case
        # Convert scores to binary predictions (0 or 1)

        predicted = (scores > threshold).int()
    else:
        # Multi-class classification case
        _, predicted = torch.max(scores, 1)
        #print(scores)

    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def torch_training_phase(model, train_loader, eval_loader, epochs, criterion, optimizer, patience, plot=True, save_best=True, save_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = float('-inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            scores = model(inputs)
            #scores = scores.to(device)

            #print(scores.shape,scores.data[0])
            if isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                labels = labels.float()
                if len(scores.shape)>1:
                    scores = scores.squeeze(1)

            #add batch size if single item
            #if len(scores.shape)<2:
            #    scores = scores.unsqueeze(0)
            #print("sc",scores,"lab",labels)
            loss = criterion(scores, labels)
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
        with torch.no_grad():
            for inputs, labels in tqdm(eval_loader, desc=f"Epoch {epoch + 1}/{epochs} [Evaluation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                scores = model(inputs)
                if isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    labels = labels.float()
                    if len(scores.shape)>1:
                        scores = scores.squeeze(1)

                #add batch size if single item
                #if len(scores.shape)<2:
                #    scores = scores.unsqueeze(0)
                loss = criterion(scores, labels)
                total_eval_loss += loss.item()
                total_eval_accuracy += calculate_accuracy(scores, labels)

        avg_eval_loss = total_eval_loss / len(eval_loader)
        avg_eval_accuracy = total_eval_accuracy / len(eval_loader)
        val_losses.append(avg_eval_loss)
        val_accuracies.append(avg_eval_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_eval_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}, Validation Accuracy: {avg_eval_accuracy:.4f}')

        if avg_eval_accuracy > best_val_accuracy:
            print(f"Improved val accuracy from {best_val_accuracy} to {avg_eval_accuracy}")
            best_val_accuracy = avg_eval_accuracy
            epochs_no_improve = 0
            best_model_state = model.state_dict()

            if save_best:
                torch.save(best_model_state, f'{save_name}.pt')
                print(f"Model saved as {save_name}.pt")


        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve == patience:
            epochs = epoch+1
            print('Early stopping triggered')
            break

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xlim(1, epochs)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim(1, epochs)
        plt.ylim(0, 3)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Load the best model state before returning
    model.load_state_dict(best_model_state)
    return model


def torch_test_phase(model,test_loader,criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    total_test_loss, total_test_accuracy = 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the correct device

            labels = labels.clone().detach()

            scores = model(inputs)

            if isinstance(criterion,nn.BCELoss) or isinstance(criterion,nn.BCEWithLogitsLoss):
                labels = labels.float()

                if len(scores.shape)>1:
                    scores = scores.squeeze(1)

            #add batch if single item
            #if len(scores.shape)<2:
            #    scores = scores.unsqueeze(0)

            loss = criterion(scores, labels)

            total_test_loss += loss.item()
            total_test_accuracy += calculate_accuracy(scores, labels)

    avg_test_loss = round(total_test_loss / len(test_loader),3)
    avg_test_accuracy = round(total_test_accuracy / len(test_loader),3)
    print(f'Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy:}')

    return avg_test_loss,avg_test_accuracy

def get_predictions_and_targets(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            #print(output)
            pred = output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability as the prediction
            predictions.extend(pred.view_as(target).cpu().numpy())
            targets.extend(target.cpu().numpy())
    #print(predictions,targets)
    return predictions, targets

def plot_confusion_matrix(model, loader,  class_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictions, targets = get_predictions_and_targets(model, loader)
    # Compute the confusion matrix
    cm = confusion_matrix(targets, predictions,normalize='true')
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(5,5))  # Adjust the size as needed
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.3f', square=True)
    # Labels, title, and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)
    plt.tight_layout()
    plt.show()

#official pytorch method
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()

    # Reshape waveform if it's 1D (single channel)
    if len(waveform.shape) == 1:
        waveform = waveform.reshape(1, -1)

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

#official pytorch method modified
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None, axes=None):
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()

    # Reshape waveform if it's 1D (single channel)
    if len(waveform.shape) == 1:
        waveform = waveform.reshape(1, -1)

    num_channels, _ = waveform.shape

    if axes is None:
        figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    axes[0].set_title(title)  # Set title for the first axes if multiple channels



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import Audio, display
import torch

def plot_waveform_and_specgram(sample, sample_rate, title="Sample"):
    plot_waveform(sample, sample_rate, title=f"{title} Waveform")
    plot_specgram(sample, sample_rate, title=f"{title} Spectrogram")
    plt.show()

def compare_samples(sample1, sample2, sample_rate, title1="Sample 1", title2="Sample 2"):

    if isinstance(sample1, torch.Tensor):
        if sample1.is_cuda:
            sample1 = sample1.cpu()
        sample1 = sample1.numpy()

    if isinstance(sample2, torch.Tensor):
        if sample2.is_cuda:
            sample2 = sample2.cpu()
        sample2 = sample2.numpy()

    difference_waveform = sample1 - sample2

    # Plot waveforms
    plt.figure(figsize=(14, 7))

    # Waveform 1
    plt.subplot(2, 3, 1)
    plt.plot(np.linspace(0, 1, len(sample1)), sample1, label=title1)
    plt.title(f"{title1} Waveform")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Waveform 2
    plt.subplot(2, 3, 2)
    plt.plot(np.linspace(0, 1, len(sample2)), sample2, label=title2)
    plt.title(f"{title2} Waveform")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Waveform Difference
    plt.subplot(2, 3, 3)
    plt.plot(np.linspace(0, 1, len(sample2)), sample2, 'r', label=title2, alpha=0.5)  # Second sample in red
    plt.plot(np.linspace(0, 1, len(sample1)), sample1, 'b', label=title1, alpha=0.5)  # First sample in blue
    plt.title("Waveform Differences")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Spectrogram 1
    plt.subplot(2, 3, 4)
    plot_specgram(sample1, sample_rate, title=f"{title1} Spectrogram", axes=plt.gca())

    # Spectrogram 2
    plt.subplot(2, 3, 5)
    plot_specgram(sample2, sample_rate, title=f"{title2} Spectrogram", axes=plt.gca())

    # Spectrogram Difference
    plt.subplot(2, 3, 6)
    difference_waveform = sample1 - sample2
    plot_specgram(difference_waveform, sample_rate, title="Spectrogram Differences", axes=plt.gca())
    plt.tight_layout()
    plt.show()

    # Third row: Display the audios and the audio difference
    print("Sample 1 Audio:")
    display(Audio(data=sample1, rate=sample_rate))
    print("Sample 2 Audio:")
    display(Audio(data=sample2, rate=sample_rate))
    print("Audio Difference:")
    display(Audio(data=difference_waveform, rate=sample_rate))




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



