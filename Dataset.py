import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
import shutil
import math
import torchaudio
from torch import nn, optim
from tqdm import tqdm


class SpReWDataset(Dataset):
    def __init__(self, base_path,transform=None, target_transform=None, generate_signals = False):
        self.base_path = base_path
        self.categories = ['C00', 'C01', 'W01', 'W02']

        speakers = sorted([ int(speaker) for speaker in (os.listdir(os.path.join(self.base_path,self.categories[0])))])
        self.authorized_speakers = ['000'+str(speaker) for speaker in speakers[:10]] #from 0000 to 0009
        self.unauthorized_speakers = ['00'+str(speaker) for speaker in speakers[10:]] #from 0010 to 0019
        print(f" Authorized speakers: {self.authorized_speakers} \n Unauthorized speakers: {self.unauthorized_speakers}")
        self.speakers = self.authorized_speakers+self.unauthorized_speakers
        self.int_to_speakers = {int(key):value for key,value in enumerate(self.speakers)}
        self.speakers_to_int = {value:int(key) for key,value in self.int_to_speakers.items()}


        self.audio_paths = self.get_audio_paths()
        if generate_signals:
            self.signals = self.generate_signals(self.audio_paths.keys())
        self.splitted_paths = {'training': [], 'validation': [], 'test': []}  # Initialize as a dictionary
        self.transform = transform
        self.target_transform = target_transform


    def get_base_path(self):
        return base_path


    def get_speakers(self):
        return self.speakers


    def get_authorized_speakers(self):
        return self.authorized_speakers


    def get_unauthorized_speakers(self):
        return self.unauthorized_speakers


    def get_categories(self):
        return self.categories


    def get_speaker_from_path(self,path):
        base_name = os.path.basename(path)
        speaker = base_name.split("_")[1]
        return speaker


    def get_category_from_path(self,path):
        base_name = os.path.basename(path)
        category = base_name.split("_")[2]
        return category


    def get_label_from_path(self,path):
        speaker = self.get_speaker_from_path(path)
        label = self.speakers_to_int[speaker]
        return label

    def get_signal(self,path):
        return self.signals[path]

    def generate_signals(self,paths):
        signals = {}
        for path in paths:
            if self.is_audio_file(path):
                signals[path] = torchaudio.load(path)
        return signals

    def is_audio_file(self, file_path):
        # Add other audio file extensions as needed
        return file_path.lower().endswith(('.wav', '.mp3', '.flac', '.aac'))



    def print_info(self):
        print(f"SpReW object from base_path: {self.base_path} of length {len(self)}")

    def get_audio_paths(self):
        self.audio_paths = {}

        for category in self.categories:
            for speaker in os.listdir(os.path.join(self.base_path,category)):
                for audio in os.listdir(os.path.join(self.base_path,category,speaker)):

                    audio_path = os.path.join(self.base_path,category,speaker,audio)
                    label = self.speakers_to_int[speaker]
                    read_speaker = self.get_speaker_from_path(audio_path)
                    read_cat = self.get_category_from_path(audio_path)

                    if read_cat!=category:
                        print("warning! wrong category in:",audio_path)
                        basename = os.path.basename(audio_path)
                        new_basename = basename.replace(read_cat,category)
                        new_path = os.path.join(self.base_path, category, speaker, new_basename)
                        # Rename the file
                        shutil.move(audio_path, new_path)
                        print(f"Renamed '{audio_path}' to '{new_path}'")
                        # Update the path to the new file
                        audio_path = new_path

                    if read_speaker!=speaker:
                        print("warning! wrong label in:",audio_path)
                        basename = os.path.basename(audio_path)
                        new_basename = basename.replace(str(read_speaker),str(speaker))
                        new_path = os.path.join(self.base_path, category, speaker, new_basename)
                        # Rename the file
                        shutil.move(audio_path, new_path)
                        print(f"Renamed '{audio_path}' to '{new_path}'")
                        # Update the path to the new file
                        audio_path = new_path

                    self.audio_paths[audio_path] = [label,category]

        return self.audio_paths

    def set_audio_paths(self,paths):
        self.audio_paths = paths


    def get_splitted_paths(self):
        return self.splitted_paths

    def set_splitted_paths(self,paths):
        self.splitted_paths = paths


    def split_paths(self,split_training,split_validation,split_test):
        temp_test = split_test/(split_test+split_validation)
        for category in self.categories:
            for speaker in os.listdir(os.path.join(self.base_path,category)):
                speaker_path = os.path.join(self.base_path,category,speaker)
                # Label each sample based on the directory it is located in
                labeled_samples = [(os.path.join(speaker_path, audio_basename), self.speakers_to_int[speaker]) for audio_basename in os.listdir(speaker_path)]

                # Split samples into training, validation, and test

                training, temp = train_test_split(labeled_samples, test_size=split_validation+split_test)
                validation, test = train_test_split(temp, test_size=temp_test)
                self.splitted_paths['training'].extend([sample_path for sample_path,_ in training])
                self.splitted_paths['validation'].extend([sample_path for sample_path,_ in validation])
                self.splitted_paths['test'].extend([sample_path for sample_path,_ in test])

        return self.splitted_paths['training'],self.splitted_paths['validation'],self.splitted_paths['test']




    def paths_filtering(self,speakers,splits,categories,max=None):
        paths = []
        for split in splits:
            for path in self.splitted_paths[split]:

                if self.get_speaker_from_path(path) in speakers and self.get_category_from_path(path) in categories:
                    paths.append(path)
                    if max is not None and len(paths)==max:
                        return paths
        return paths


    def count_samples_filtering(self,speakers,splits,categories):
        count = 0
        for split in splits:
            for path in self.splitted_paths[split]:
                if self.get_speaker_from_path(path) in speakers and self.get_category_from_path(path) in categories:
                    count=count+1
        return count


    def show_audio_samples(self,paths,verbose=False):
        for path in paths:
            if verbose:
                print(f"Audio of speaker: {self.get_speaker_from_path(path)} category:{self.get_category_from_path(path)}")
            display(Audio(filename=path))


    def plot_samples_distribution(self):
        if self.splitted_paths is not None:
            sections = list(self.splitted_paths.keys())
            counts = [len(self.splitted_paths[section]) for section in sections]

            plt.figure(figsize=(6, 6))
            bars = plt.bar(sections, counts, color=['blue', 'red', 'green'])
            plt.ylabel('Number of samples')
            plt.title('Samples distribution')

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

            plt.show()
        else:
            print("paths not splitted, use split_paths()")




    def plot_speakers_distribution(self):

        # Count data for each set and authorization type
        training_vals_auth = [self.count_samples_filtering([speaker],["training"],self.categories) for speaker in self.authorized_speakers]+[0]*10
        training_vals_unauth = [0]*10+[self.count_samples_filtering([speaker],["training"],self.categories) for speaker in self.unauthorized_speakers]
        validation_vals_auth = [self.count_samples_filtering([speaker],["validation"],self.categories) for speaker in self.authorized_speakers]+[0]*10
        validation_vals_unauth =  [0]*10+[self.count_samples_filtering([speaker],["validation"],self.categories) for speaker in self.unauthorized_speakers]
        test_vals_auth = [self.count_samples_filtering([speaker],["test"],self.categories) for speaker in self.authorized_speakers]+[0]*10
        test_vals_unauth =  [0]*10+[self.count_samples_filtering([speaker],["test"],self.categories) for speaker in self.unauthorized_speakers]

        ind = np.arange(len(self.speakers))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(15, 10))
        fig.patch.set_facecolor('white')  # Set the background color of the figure to black
        ax.set_facecolor('white')  # Set the background color of the axes to black


        # Stacking bars for training, validation, and test
        p1 = ax.bar(ind, training_vals_auth, width, label='Training - Authorized', color='lightblue')
        p2 = ax.bar(ind, validation_vals_auth, width, bottom=training_vals_auth, label='Validation - Authorized', color='blue')
        p3 = ax.bar(ind, test_vals_auth, width, bottom=np.array(training_vals_auth) + np.array(validation_vals_auth), label='Test - Authorized', color='darkblue')

        p4 = ax.bar(ind, training_vals_unauth, width, bottom=np.array(training_vals_auth) + np.array(validation_vals_auth) + np.array(test_vals_auth), label='Training - Unauthorized', color='lightcoral')
        p5 = ax.bar(ind, validation_vals_unauth, width, bottom=np.array(training_vals_auth) + np.array(validation_vals_auth) + np.array(test_vals_auth) + np.array(training_vals_unauth), label='Validation - Unauthorized', color='red')
        p6 = ax.bar(ind, test_vals_unauth, width, bottom=np.array(training_vals_auth) + np.array(validation_vals_auth) + np.array(test_vals_auth) + np.array(training_vals_unauth) + np.array(validation_vals_unauth), label='Test - Unauthorized', color='darkred')

        for p in [p1,p2,p3,p4,p5,p6]:
            for bar in p:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, '%d' % int(height), ha='center', va='bottom')

        ax.set_ylabel('Number of Samples',color="black")
        ax.set_title('Number of Samples per Speaker in Training, Validation, and Test',color="black")
        ax.set_xticks(ind)
        ax.set_xticklabels(self.speakers, rotation=45, ha='right')
        # Set the tick colors to white
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')

        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color('black')

        plt.tight_layout()
        plt.show()



    def plot_category_distribution(self):
        labels_categories = self.categories  # Assuming self.categories is ['C00', 'C01', 'W01', 'W02']
        barWidth = 0.15
        # Positions for the bars on the x-axis
        r_positions = [np.arange(len(labels_categories)) + barWidth*i for i in range(6)]

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')  # Optional: only if you want a black background
        ax.set_facecolor('white')  # Optional: only if you want a black background

        for i, cat in enumerate(labels_categories):
            counts_train = self.count_samples_filtering(self.authorized_speakers, ["training"], [cat])
            counts_val = self.count_samples_filtering(self.authorized_speakers, ["validation"], [cat])
            counts_test = self.count_samples_filtering(self.authorized_speakers, ["test"], [cat])
            counts_train_unauth = self.count_samples_filtering(self.unauthorized_speakers, ["training"], [cat])
            counts_val_unauth = self.count_samples_filtering(self.unauthorized_speakers, ["validation"], [cat])
            counts_test_unauth = self.count_samples_filtering(self.unauthorized_speakers, ["test"], [cat])

            # Create bars for each category and dataset
            ax.bar(r_positions[0][i], counts_train, width=barWidth, label='Training Authorized' if i == 0 else "", color='blue')
            ax.text(r_positions[0][i], counts_train, '%d' % counts_train, ha='center', va='bottom')
            ax.bar(r_positions[1][i], counts_val, width=barWidth, label='Validation Authorized' if i == 0 else "", color='lightblue')
            ax.text(r_positions[1][i], counts_val, '%d' % counts_val, ha='center', va='bottom')
            ax.bar(r_positions[2][i], counts_test, width=barWidth, label='Test Authorized' if i == 0 else "", color='darkblue')
            ax.text(r_positions[2][i], counts_test, '%d' % counts_test, ha='center', va='bottom')
            ax.bar(r_positions[3][i], counts_train_unauth, width=barWidth, label='Training Unauthorized' if i == 0 else "", color='red')
            ax.text(r_positions[3][i], counts_train_unauth, '%d' % counts_train_unauth, ha='center', va='bottom')
            ax.bar(r_positions[4][i], counts_val_unauth, width=barWidth, label='Validation Unauthorized' if i == 0 else "", color='lightsalmon')
            ax.text(r_positions[4][i], counts_val_unauth, '%d' % counts_val_unauth, ha='center', va='bottom')
            ax.bar(r_positions[5][i], counts_test_unauth, width=barWidth, label='Test Unauthorized' if i == 0 else "", color='darkred')
            ax.text(r_positions[5][i], counts_test_unauth, '%d' % counts_test_unauth, ha='center', va='bottom')



        # Set labels and titles with contrasting color if using black background
        ax.set_ylabel('Number of Samples', color='white' if fig.get_facecolor() == 'black' else 'black')
        ax.set_title('Number of Samples by Category and Set', color='white' if fig.get_facecolor() == 'black' else 'black')

        # Positioning and labeling the x-ticks
        ax.set_xticks([r + 2.5*barWidth for r in range(len(labels_categories))])
        ax.set_xticklabels(labels_categories, color='white' if fig.get_facecolor() == 'black' else 'black')

        ax.legend()
        plt.tight_layout()
        plt.show()


    def __len__(self):
        return len(self.audio_paths)


    def __getitem__(self, idx):

        path = list(self.audio_paths.keys())[idx]
        signal,fs = torchaudio.load(path)
        label = self.get_label_from_path(path)
        label = torch.tensor(label, dtype=torch.long)

        # Normalize and ensure mono audio
        signal = signal / signal.abs().max()
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)


        return signal, label


class ChunkedSpReWDataset(SpReWDataset):

    def __init__(self, base_path, chunk_size=0.2, transform=None, target_transform=None, generate_signals=False):
        self.audio_paths = None
        super().__init__(base_path, transform, target_transform, generate_signals)

        self.generate_signals = generate_signals
        self.chunk_size = chunk_size
        self.signals = {}
        # Splitting the original dataset
        self.split_paths(split_training=0.7, split_validation=0.15, split_test=0.15)
        # Creating chunked paths for train, val, and test
        self.chunked_paths = {
            'training': self.divide_and_save_chunks(self.splitted_paths['training']),
            'validation': self.divide_and_save_chunks(self.splitted_paths['validation']),
            'test': self.divide_and_save_chunks(self.splitted_paths['test'])
        }

        self.splitted_paths = self.chunked_paths
        self.set_splitted_paths(self.splitted_paths)

        self.audio_paths = {}

        for section in self.chunked_paths.keys():
            for path in self.chunked_paths[section]:

                label = self.get_label_from_path(path)
                category = self.get_category_from_path(path)
                self.audio_paths[path] = [label,category]

        self.set_audio_paths(self.audio_paths)


    def get_audio_paths(self):
        if self.audio_paths is None:
            self.audio_paths = super().get_audio_paths()
            return self.audio_paths
        else:
           return self.audio_paths


    def get_splitted_paths(self):
        return self.splitted_paths


    def get_chunked_paths(self):
        return self.chunked_paths


    def get_signal(self,path):
        #print(self.signals)
        return self.signals[path]


    def divide_and_save_chunks(self, paths):
        chunked_paths = []

        for audio_path in tqdm(paths, desc="Processing audio files"):
            if not self.is_audio_file(audio_path):
                continue  # Skip if the file is not an audio file

            basename = os.path.basename(audio_path).split(".")[0]
            save_path = os.path.join(self.base_path, 'chunked_audios_' + str(self.chunk_size).replace(".","_"), basename)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            signal, fs = torchaudio.load(audio_path)
            num_samples_per_chunk = int(fs * self.chunk_size)
            num_chunks = math.ceil(signal.shape[1] / num_samples_per_chunk)

            if len(os.listdir(save_path)) >= (num_chunks - 1) and not self.generate_signals:
                #print(f"skipping {save_path},already exists")
                for chunk_path in os.listdir(save_path):
                    chunk_total_path = os.path.join(save_path, chunk_path)
                    chunked_paths.append(chunk_total_path)
                    if self.generate_signals:
                        chunk_signal,fs = torchaudio.load(chunk_total_path)
                        self.signals[chunk_total_path] = chunk_signal

                continue

            for chunk in range(num_chunks):
                start = chunk * num_samples_per_chunk
                end = start + num_samples_per_chunk

                if end > signal.shape[1]:
                    continue  # Skip saving shorter chunks

                chunk_signal = signal[:, start:end]
                chunk_file_name = f"{basename}_chunk_{chunk}.wav"
                chunk_save_path = os.path.join(save_path, chunk_file_name)
                torchaudio.save(chunk_save_path, chunk_signal, fs)
                if self.generate_signals:
                    self.signals[chunk_save_path] = chunk_signal
                chunked_paths.append(chunk_save_path)

        return chunked_paths

    def plot_samples_distribution(self):
        if self.chunked_paths:
            sections = list(self.chunked_paths.keys())
            counts = [len(self.chunked_paths[section]) for section in sections]

            plt.figure(figsize=(6, 6))
            bars = plt.bar(sections, counts, color=['blue', 'red', 'green'])
            plt.ylabel('Number of chunks')
            plt.title('Chunks distribution')

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom', ha='center')

            plt.show()
        else:
            print("No chunked paths available.")

    def __len__(self):
        return sum(len(chunks) for chunks in self.chunked_paths.values())

    def __getitem__(self, idx):
        for dataset_type, chunks in self.chunked_paths.items():
            if idx < len(chunks):
                audio_path = chunks[idx]
                signal, fs = torchaudio.load(audio_path)
                label = self.get_label_from_path(audio_path)

                # Normalize and ensure mono audio
                signal = signal / signal.abs().max()
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0, keepdim=True)

                label = torch.tensor(label, dtype=torch.long)
                return signal, label

            idx -= len(chunks)


class EmbeddingDatasetSI(Dataset):
    def __init__(self, embeddings,labels):

        self.embeddings = embeddings
        self.labels = labels

        #print(self.embeddings,self.labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EmbeddingDatasetSV(Dataset):
    def __init__(self, embeddings,authorized_labels):

        self.embeddings = [emb[0] for emb in embeddings]
        self.labels = [1 if emb[2] in authorized_labels else 0 for emb in embeddings]

        #print(self.embeddings,self.labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
        
        
class PathsDatasetSI(Dataset):
    def __init__(self, paths,sprew):
        self.paths = [path for path in paths]
        self.labels = [sprew.get_label_from_path(path) for path in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx], self.labels[idx]


class PathsDatasetSV(Dataset):
    def __init__(self, paths,sprew,authorized_labels):

        self.paths = [path for path in paths]
        self.labels = [sprew.get_label_from_path(path) for path in paths]
        self.labels = [1 if label in authorized_labels else 0 for label in self.labels]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx], self.labels[idx]


class SignalsSI(Dataset):
    def __init__(self, paths, sprew, to_tensor=False):
        self.to_tensor = to_tensor
        self.signals = [sprew.get_signal(path) for path in paths]
        self.labels = [sprew.get_label_from_path(path) for path in paths]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal, label = self.signals[idx], self.labels[idx]

        if self.to_tensor:
            # Assuming the signal is a numpy array or a similar structure that can be directly converted to a tensor
            signal = torch.tensor(signal, dtype=torch.float32)

        return signal, label


class SignalsSV(Dataset):
    def __init__(self, paths, sprew, authorized_labels, to_tensor=False):
        self.to_tensor = to_tensor
        self.signals = [sprew.get_signal(path) for path in paths]
        self.labels = [sprew.get_label_from_path(path) for path in paths]
        self.labels = [1 if label in authorized_labels else 0 for label in self.labels]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal, label = self.signals[idx], self.labels[idx]

        if self.to_tensor:
            # Assuming the signal is a numpy array or a similar structure that can be directly converted to a tensor
            signal = torch.tensor(signal, dtype=torch.float32)

        return signal, label
    
import librosa
import numpy as np

class MelSpectogramSI(Dataset):
    def __init__(self, paths, sprew, to_tensor=False, to_db=True, n_mels=80, sr=16000):
        self.sr = sr
        self.n_mels = n_mels
        self.to_tensor = to_tensor
        self.signals = [sprew.get_signal(path) for path in paths]

        # Ensure signals are numpy arrays
        self.signals = [np.array(signal) if not isinstance(signal, np.ndarray) else signal for signal in self.signals]

        self.spectograms = [librosa.feature.melspectrogram(y=signal, sr=self.sr, n_mels=self.n_mels) for signal in self.signals]

        if to_db:
            self.spectograms = [librosa.power_to_db(mel_spec, ref=np.max) for mel_spec in self.spectograms]

        self.labels = [sprew.get_label_from_path(path) for path in paths]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        spectrogram, label = self.spectograms[idx], self.labels[idx]

        if self.to_tensor:
            # Convert spectrogram to tensor and add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        return spectrogram, label
    
class MelSpectogramSV(Dataset):
    def __init__(self, paths, sprew, authorized_labels, to_tensor=False, to_db=True, n_mels=80, sr=16000):
        self.sr = sr
        self.n_mels = n_mels
        self.to_tensor = to_tensor
        self.signals = [sprew.get_signal(path) for path in paths]

        # Ensure signals are numpy arrays
        self.signals = [np.array(signal) if not isinstance(signal, np.ndarray) else signal for signal in self.signals]

        self.spectograms = [librosa.feature.melspectrogram(y=signal, sr=self.sr, n_mels=self.n_mels) for signal in self.signals]

        if to_db:
            self.spectograms = [librosa.power_to_db(mel_spec, ref=np.max) for mel_spec in self.spectograms]

        self.labels = [sprew.get_label_from_path(path) for path in paths]
        self.labels = [1 if label in authorized_labels else 0 for label in self.labels]


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        spectrogram, label = self.spectograms[idx], self.labels[idx]

        if self.to_tensor:
            # Convert spectrogram to tensor and add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        return spectrogram, label