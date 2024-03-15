#@title ART Attacks classes

import os
import pickle
import matplotlib.pyplot as plt
from art.estimators.classification import SklearnClassifier
import itertools
from itertools import product
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import DeepFool
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import random
import colorsys
import numpy as np
import torch
import librosa.display
from tqdm import tqdm
import random
from torch.nn.functional import softmax
from Dataset import *
from Evaluation import *

LOWER = -1
UPPER = 1

def save_dict_to_pickle(data, filepath):
    """
    Save a dictionary to a pickle file.

    Args:
        data (dict): Dictionary to be saved.
        filepath (str): Path to the pickle file.

    Returns:
        None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_dict_from_pickle(filepath):
    """
    Load a dictionary from a pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        dict: Loaded dictionary.
    """
    try:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
    except FileNotFoundError:
        print(f'File {filepath} does not exist')
        loaded_data = {}
    except EOFError:
        print(f'File {filepath} is empty')
        loaded_data = {}

    return loaded_data




class ART_Attacks:
    """
    Class for conducting adversarial attacks using the ART library.
    """

    def __init__(self, classifier):
        """
        Initialize the ART_Attacks class.

        Args:
            classifier: The classifier model used for attacks.
        """
        self.classifier = classifier
        self.attacker = None
        self.dict_path = None
        self.model = None

    def classifier_to_attack(self, clip_values, x, y):
        """
        Convert a classifier to an attack model.

        Args:
            clip_values: The minimum and maximum values of the input features.
            x: The input data.
            y: The labels.

        Returns:
            The converted attack model.
        """
        # Convert the classifier to an attack model
        return self.classifier

    def set_params(self, params):
        """
        Set parameters for the attacker.

        Args:
            params: Parameters for the attacker.

        Returns:
            None
        """
        self.attacker.set_params(**params)
        self.attacker.set_params(**{'verbose': False})

    def attack(self, samples_to_attack, target_labels=None):
        """
        Generate adversarial samples.

        Args:
            samples_to_attack: The samples to attack.
            target_labels: The target labels for targeted attacks.

        Returns:
            Adversarial samples.
        """
        return samples_to_attack

    def top_attacks(self, model, combinations, samples_labels, samples_to_attack, N_top=10, scaler=None):
        """
        Find the top attacks based on accuracy.

        Args:
            model: The classifier model.
            combinations: The combinations of attack parameters.
            samples_labels: The labels of the input samples.
            samples_to_attack: The samples to attack.
            N_top: The number of top attacks to return.
            scaler: A scaler object for preprocessing.

        Returns:
            Top attack combinations and samples.
        """
        top_combinations = []
        top_samples = []

        return top_combinations, top_samples

    def is_one_hot(self, array):
        """
        Check if an array is one-hot encoded.

        Args:
            array: The input array.

        Returns:
            True if the array is one-hot encoded, False otherwise.
        """
        if array.ndim != 2:
            return False
        for row in array:
            if np.sum(row) != 1 or np.any((row != 0) & (row != 1)):
                return False
        return True

    @staticmethod
    def generate_parameters_dicts(params):
        """
        Generate permutations of parameters.

        Args:
            params: A dictionary of parameters.

        Returns:
            A list of parameter dictionaries.
        """
        keys, values = zip(*params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts








class ART_FGSM(ART_Attacks):
    """
    Subclass representing the Fast Gradient Sign Method (FGSM) attack.

    Inherits from `ART_Attacks`.
    """

    def __init__(self, classifier):
        """
        Initialize the ART_FGSM class.

        Args:
            classifier: The classifier model used for attacks.
        """
        super().__init__(classifier)
        self.attacker = FastGradientMethod(estimator=classifier)


class ART_BIM(ART_Attacks):
    """
    Subclass representing the Basic Iterative Method (BIM) attack.

    Inherits from `ART_Attacks`.
    """

    def __init__(self, classifier):
        """
        Initialize the ART_BIM class.

        Args:
            classifier: The classifier model used for attacks.
        """
        super().__init__(classifier)
        self.attacker = BasicIterativeMethod(estimator=classifier)


class ART_PGD(ART_Attacks):
    """
    Subclass representing the Projected Gradient Descent (PGD) attack.

    Inherits from `ART_Attacks`.
    """

    def __init__(self, classifier):
        """
        Initialize the ART_PGD class.

        Args:
            classifier: The classifier model used for attacks.
        """
        super().__init__(classifier)
        self.attacker = ProjectedGradientDescent(estimator=classifier)


class ART_DF(ART_Attacks):
    """
    Subclass representing the DeepFool attack.

    Inherits from `ART_Attacks`.
    """

    def __init__(self, classifier):
        """
        Initialize the ART_DF class.

        Args:
            classifier: The classifier model used for attacks.
        """
        super().__init__(classifier)
        self.attacker = DeepFool(classifier)


class ART_CW(ART_Attacks):
    """
    Subclass representing the Carlini and Wagner L2 (CW) attack.

    Inherits from `ART_Attacks`.
    """

    def __init__(self, classifier):
        """
        Initialize the ART_CW class.

        Args:
            classifier: The classifier model used for attacks.
        """
        super().__init__(classifier)
        self.attacker = CarliniL2Method(classifier)




class MultipleARTAttacks:
    def __init__(self,model,attacks, combinations, samples, labels):
        """
        Initialize the MultipleARTAttacks class.

        :param model: The neural network model.
        :param attacks: List of ART_Attacks instances.
        :param combinations: Dictionary with attack names as keys and lists of parameter combinations as values.
        :param samples: The input samples to attack.
        :param labels: The true labels of the input samples.
        """

        self.model = model
        self.attacks = attacks
        self.combinations = combinations
        self.samples = samples
        self.labels = labels
        self.results = {}
        self.N_top = None

    def run_attacks(self,N_top,model = None):

        if model is None:
            model = self.model
        self.results[model] = {}
        self.N_top = N_top

        for attack_name, attack in self.attacks.items():

            if attack_name in self.results[model].keys():
                print("Attack already done,skipping..")
                continue

            print(f"Running attack: {attack_name}")
            attack_combinations = self.combinations[attack_name]

            top_params, top_samples = attack.top_attacks(
                self.model, attack_combinations, self.labels, self.samples, N_top=N_top
            )
            #print(top_params)
            self.results[model][attack_name] = {
                'params': top_params,
                'samples': top_samples
            }




    def get_results(self):
        return self.results

    def set_results(self,results):
        self.results = results


    def get_mean_perturbation(self,samples_a,samples_b):
        individual_perturbations = []

        for sample_a, sample_b in zip(samples_a, samples_b):
            # Convert sample_or to a numpy array if it's a torch.Tensor
            if isinstance(sample_a, torch.Tensor):
                sample_a = sample_a.cpu().detach().numpy()

            # Ensure sample_adv is a numpy array (in case it's a torch.Tensor)
            if isinstance(sample_b, torch.Tensor):
                sample_b = sample_b.cpu().detach().numpy()

            # Compute the perturbation for the current pair
            perturbation = np.mean(np.abs(sample_b-sample_a))
            individual_perturbations.append(perturbation)

        return np.mean(individual_perturbations)


    def get_effective_samples(self, attacks, top_attacks = None, samples_per_attack = None):

        device = next(self.model.parameters()).device  # Get the device the model is on
        adv_samples_per_attack = {}
        samples_params = {}

        if top_attacks is None:
            top_attacks = self.N_top

        original_samples_tensor = torch.stack([torch.Tensor(sample) for sample in self.samples])
        original_samples_tensor = original_samples_tensor.to(device)
        for attack_name, attack_dict in self.results[self.model].items():
            if attack_name not in attacks:
                continue
            adv_samples_per_attack[attack_name] = []

            all_params = attack_dict['params']
            list_adv_samples = attack_dict['samples'][:top_attacks]

            for i, adv_samples in enumerate(list_adv_samples):
                # Check shapes and types of adv_samples before converting to tensor
                params = all_params[i]
                adv_samples_tensor = torch.tensor(adv_samples, dtype=torch.float32).to(device)

                self.model.eval()

                with torch.no_grad():
                    original_predictions = self.model(original_samples_tensor).argmax(dim=1)
                    adversarial_predictions = self.model(adv_samples_tensor).argmax(dim=1)


                    # Find samples where predictions differ and original prediction is zero
                    diff_indices = torch.where((original_predictions != adversarial_predictions) & (original_predictions == 0))[0]

                    effective_advs = adv_samples_tensor[diff_indices]
                    added_samples = 0
                    # Add effective adversarial samples to the list, up to the desired number
                    for adv in effective_advs:
                        adv_samples_per_attack[attack_name].append(adv.cpu().numpy())  # Convert back to numpy if needed
                        added_samples += 1
                        samples_params[str(adv.cpu().numpy())] = str(params)

                        if samples_per_attack is not None and added_samples >= samples_per_attack:
                            break

        return adv_samples_per_attack,samples_params




    def get_random_samples(self, N):
        random_samples_per_attack = {}

        # Ensure each attack contributes equally to the total N samples
        samples_per_attack = N // len(self.results)

        for attack_name, attack_dict in self.results[self.model].items():
            # Get all samples for the current attack
            all_samples = attack_dict['samples']
            flat_samples = [item for sublist in all_samples for item in sublist]  # Flatten the list of lists

            # Randomly select samples from the flattened list
            selected_samples = random.sample(flat_samples, min(samples_per_attack, len(flat_samples)))

            # If the samples are tensors, convert them to numpy arrays
            selected_samples = [sample.cpu().numpy() if isinstance(sample, torch.Tensor) else sample for sample in selected_samples]

            random_samples_per_attack[attack_name] = selected_samples

        return random_samples_per_attack



    def analyze_results(self,model_to_analyze = None):

        
	
        if model_to_analyze is None:
            model_to_analyze = self.model

        device = next(model_to_analyze.parameters()).device

        if model_to_analyze not in self.results.keys():
            self.results[model_to_analyze] = {}


        for attack_name, result in self.results[self.model].items():
            print(f"Results for {attack_name} on {str(model_to_analyze).split('(')[0]}:")
            top_params = result['params']
            top_samples = result['samples']
            #print(top_params)

            if attack_name not in self.results[model_to_analyze].keys():
                self.results[model_to_analyze][attack_name] = {}
                self.results[model_to_analyze][attack_name]['params'] = top_params
                self.results[model_to_analyze][attack_name]['samples'] = top_samples

            self.results[model_to_analyze][attack_name]['accuracies'] = []
            self.results[model_to_analyze][attack_name]['mean_perturbations'] = []
            self.results[model_to_analyze][attack_name]['mean_snrs'] = []

            for params, samples in zip(top_params, top_samples):

                # Convert samples to a PyTorch Tensor if they're not already
                samples_tensor = torch.tensor(samples, dtype=torch.float32)
                samples_tensor = samples_tensor.to(device)

                # Ensure model is in evaluation mode
                model_to_analyze.eval()

                # Get predictions
                with torch.no_grad():
                    #print(samples_tensor.size())
                    predictions = model_to_analyze(samples_tensor)

                # Convert predictions to numpy array if necessary for further processing
                predictions = predictions.cpu().numpy()  # Remove '.cpu()' if not using CUDA
                #
                predictions = softmax(torch.from_numpy(predictions), dim=1).numpy()
                #
                predicted_labels = np.argmax(predictions, axis=1)
                accuracy = np.mean(predicted_labels == self.labels)
                snr_list = [SNR(sample,adv_sample) for adv_sample,sample in zip(samples,self.samples)]#self.calculate_snr(samples, self.samples)
                mean_snr = np.mean(snr_list)
                # Compute the overall mean perturbation
                mean_perturbation = self.get_mean_perturbation(samples,self.samples)

                print(f"Params: {params}, Accuracy: {accuracy}, Mean Perturbation: {mean_perturbation}, Mean snr: {mean_snr}")
                self.results[model_to_analyze][attack_name]['accuracies'].append(accuracy)
                self.results[model_to_analyze][attack_name]['mean_perturbations'].append(mean_perturbation)
                self.results[model_to_analyze][attack_name]['mean_snrs'].append(mean_snr)



    def plot_evaluation_curve(self,model = None):
        if model is None:
            model = self.model
        #print(self.results,self.results.keys())
        model_results = self.results[model]


        for attack_name, result in model_results.items():
            top_params = result['params']
            top_samples = result['samples']

            flattened_samples = top_samples[0].reshape(-1)  # Flatten into a single array
            max_v = np.max(flattened_samples)
            min_v = np.min(flattened_samples)
            max_pert = max_v - min_v

            accuracies = result['accuracies']
            mean_perturbations = result['mean_perturbations']


            # Pair, sort by mean perturbation in descending order, and unzip
            paired = list(zip(mean_perturbations, accuracies))
            paired.sort(reverse=True, key=lambda x: x[0])
            mean_perturbations_sorted, accuracies_sorted = zip(*paired)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(mean_perturbations_sorted, accuracies_sorted, label=f'{attack_name} Attack', marker='o')
            plt.xlabel('Mean Perturbation')
            plt.ylabel('Accuracy')
            plt.title(f'Evaluation Curve for {attack_name} Attack')
            plt.legend()
            plt.grid(True)
            #plt.tight_layout()
            # Set x-axis and y-axis limits
            #plt.xlim(0, max(mean_perturbations))
            plt.xlim(0, max_pert)
            margin = 0.1
            plt.ylim(0-margin, 1+margin)

            plt.show()


    def plot_evaluation_curve_snr(self,model = None):
        if model is None:
            model = self.model
        model_results = self.results[model]
        for attack_name, result in model_results.items():
            top_params = result['params']
            top_samples = result['samples']


            accuracies = result['accuracies']
            mean_snrs = result['mean_snrs']

            # Pair, sort by mean SNR in descending order, and unzip
            paired = list(zip(mean_snrs, accuracies))
            paired.sort(reverse=True, key=lambda x: x[0])
            mean_snrs_sorted, accuracies_sorted = zip(*paired)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(mean_snrs_sorted, accuracies_sorted, label=f'{attack_name} Attack', marker='o')
            plt.xlabel('Mean Signal-to-Noise Ratio (SNR)')
            plt.ylabel('Accuracy')
            plt.title(f'Evaluation Curve for {attack_name} Attack (Mean SNR vs. Accuracy)')
            plt.legend()
            plt.grid(True)

            mean_snrs = np.array(mean_snrs)  # Ensure it's a numpy array
            mean_snrs = mean_snrs[np.isfinite(mean_snrs)]  # Keep only finite values

            plt.xlim(-30,0)
            margin = 0.1
            plt.ylim(0-margin, 1+margin)
            plt.show()





    def plot_top_n_attacks(self, N,model = None):
        """
        For each attack, plot a separate chart where the x-axis shows the parameter combinations
        (e.g., {"eps = 5, eps_step = 1"}, {"eps = 3, eps_step = 0.5"}, ...) and the y-axis shows
        the accuracies of those combinations (e.g., 0.97, 0.95, ...).
        """
        if model is None:
            model = self.model
        model_results = self.results[model]
        all_results = {}
        for attack_name, result in model_results.items():
            top_params = result['params']
            top_samples = result['samples']
            accuracies = result['accuracies']

            attack_results = [(params,accuracy) for params,accuracy in zip(top_params,accuracies)]
            '''
            for params, samples in zip(top_params, top_samples):
                samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(samples_tensor)
                predictions = predictions.cpu().numpy()
                predicted_labels = np.argmax(predictions, axis=1)
                accuracy = np.mean(predicted_labels == self.labels)
                accuracy = round(accuracy,3)


                accuracies.append(accuracy)
                attack_results.append((params, accuracy))
            '''
            # Sort results for each attack based on accuracy
            sorted_results = sorted(attack_results, key=lambda x: x[1])[:N]
            all_results[attack_name] = sorted_results

        # Plotting
        for attack_name, sorted_results in all_results.items():
            param_combinations = [f"{', '.join(f'{k}={v}' for k, v in params.items())}" for params, _ in sorted_results]
            accuracies = [accuracy for _, accuracy in sorted_results]

            plt.figure(figsize=(10, 5))
            plt.plot(param_combinations, accuracies, color='skyblue')
            plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
            plt.xlabel('Parameter Combinations')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy of Parameter Combinations for {attack_name} Attack')
            #plt.tight_layout()  # Adjust layout to fit rotated x-axis labels
            margin = 0.1
            plt.ylim(0-margin, 1+margin)
            plt.grid(True)
            plt.show()



    def show_adversarial_samples(self, attacks, top_attacks, samples_per_attack):
        model = self.model
        device = next(model.parameters()).device  # Get the device the model is on
        model.eval()  # Set the model to evaluation mode
        sample_rate = 16000

        effective_samples,samples_params = self.get_effective_samples(attacks, top_attacks, samples_per_attack)
        original_samples_tensor = torch.stack([torch.Tensor(sample) for sample in self.samples])
        original_samples_tensor = original_samples_tensor.to(device)

        for attack_name, adv_samples in effective_samples.items():
            adv_samples_tensor = torch.tensor(adv_samples, dtype=torch.float32).to(device)

            with torch.no_grad():
                original_predictions = self.model(original_samples_tensor).argmax(dim=1)
                adversarial_predictions = self.model(adv_samples_tensor).argmax(dim=1)

            for i, (original_prediction, adversarial_prediction) in enumerate(zip(original_predictions, adversarial_predictions)):

                if original_prediction == 1 or original_prediction == adversarial_prediction:
                    continue
                params = samples_params[str(adv_samples[i])]
                original_sample = self.samples[i].squeeze()
                adv_sample = adv_samples[i].squeeze()
                print(f"Attack :{attack_name} params: {params}")
                compare_samples(original_sample,adv_sample,sample_rate, title1=f"Original sample (Prediction {original_prediction.item()})", title2=f"{attack_name} Adversarial sample (Prediction {adversarial_prediction.item()})")



    def compare_models(self, models=None, models_name={}):
        model_accuracies = {}
        model_params = {}

        if models is None:
            models = self.results.keys()

        # Iterate through each model and their results
        for model_name, model_results in self.results.items():
            if model_name not in models:
                continue

            if model_name not in model_accuracies:
                model_accuracies[model_name] = {}
                model_params[model_name] = {}

            for attack_name, result in model_results.items():
                # Store accuracies and a string representation of parameters for each attack
                model_accuracies[model_name][attack_name] = result['accuracies']
                model_params[model_name][attack_name] = [', '.join([f'{k}={v}' for k, v in param.items()]) for param in result['params']]

        # Determine the number of attacks to plot
        num_attacks = len(set(k for d in model_accuracies.values() for k in d))
        fig, axes = plt.subplots(1, num_attacks, figsize=(14 * num_attacks, 8))  # Adjust the total width as needed

        # Ensure axes is an array even with a single subplot
        if num_attacks == 1:
            axes = [axes]

        # Plotting for each attack
        for ax, attack_name in zip(axes, sorted(set(k for d in model_accuracies.values() for k in d))):
            colors = plt.cm.jet(np.linspace(0, 1, len(model_accuracies)))
            line_styles = ['-', '--', '-.', ':']  # Different line styles for differentiation
            marker_types = ['o', 's', '^', 'p', '*']  # Different markers for further differentiation

            for model_idx, (model_name, attacks) in enumerate(model_accuracies.items()):
                if attack_name in attacks:
                    accs = attacks[attack_name]
                    params = model_params[model_name][attack_name]
                    if model_name not in models_name.keys():
                        label = str(model_name).split("(")[0]
                    else:
                        label = models_name[model_name]

                    # Choose line style and marker based on model index
                    line_style = line_styles[model_idx % len(line_styles)]
                    marker_type = marker_types[model_idx % len(marker_types)]

                    ax.plot(params, accs, label=label, marker=marker_type, linestyle=line_style, color=colors[model_idx])

            ax.set_xticklabels(params, rotation=90, ha="right", fontsize='x-small')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Parameters')
            ax.set_title(f'Accuracy Comparison for {attack_name}')
            ax.legend()
            ax.grid(True)
            margin = 0.1
            ax.set_ylim(0 - margin, 1 + margin)

        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
# Commented out IPython magic to ensure Python compatibility.
#@title SG Attacks classes

import itertools
import torch
import numpy as np
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score


# %cd "/content/drive/MyDrive/MAGISTRALE/TESI_LAUREA_MAGISTRALE/SpeakerGuard"


from SpeakerGuard.attack.FAKEBOB import FAKEBOB
from SpeakerGuard.attack.FGSM import FGSM
from SpeakerGuard.attack.PGD import PGD
from SpeakerGuard.attack.CW2 import CW2
from SpeakerGuard.attack.CWinf import CWinf
from SpeakerGuard.attack.Kenan import Kenan
from SpeakerGuard.attack.SirenAttack import SirenAttack


LOWER = -1
UPPER = 1


class SG_Attack:

    def __init__(self):
        pass

    def attack(self, x, y):
        raise NotImplementedError

    def attack_batch(self, x_batch, y_batch, **kwargs):
        raise NotImplementedError

    def attack_dataloader(self, dataloader):
        success_counter = 0
        adversarial_samples = []
        perturbations = []

        for i, (x_batch, y_batch) in enumerate(tqdm(dataloader, desc="Attacking")):
            adv_x, success = self.attack_batch(x_batch, y_batch, batch_id=i)
            success_counter += int(success[0])
            adversarial_samples.append(adv_x)

            if isinstance(adv_x, np.ndarray):
                adv_x = torch.from_numpy(adv_x).to(x_batch.device)
            #perturbation = np.linalg.norm(adv_x - x_batch_np)
            # Ensure both tensors are on the same device and have the same shape
            assert adv_x.device == x_batch.device, "Tensors are on different devices."
            assert adv_x.shape == x_batch.shape, f"Shapes are different: {adv_x.shape} vs {x_batch.shape}"

            # Detach tensors from the computation graph if necessary
            adv_x_detached = adv_x.detach()
            x_batch_detached = x_batch.detach()

            # Move to CPU and convert to numpy
            adv_x_np = adv_x_detached.cpu().numpy()
            x_batch_np = x_batch_detached.cpu().numpy()

            # Compute perturbation norm
            perturbation = np.linalg.norm(adv_x_np - x_batch_np)

            perturbations.append(perturbation)

        success_rate = success_counter / (i + 1)
        return adversarial_samples, success_rate,perturbations

    def preprocess_batch(self, x_batch, y_batch):

        while len(x_batch.size()) > 2:
            x_batch = x_batch.squeeze(0)

        x_reshaped = x_batch.clone().detach().unsqueeze(0)
        x_min = x_reshaped.min()
        x_max = x_reshaped.max()
        x_reshaped_normalized = 2 * ((x_reshaped - x_min) / (x_max - x_min)) - 1
        return x_reshaped_normalized, y_batch

    def attacks_combs(self, dataloader, params):
        self.parameters_combinations = self.parameters_permutation(params)
        self.comb_success = {}
        print(f"Total comb:{len(self.parameters_combinations)}")
        for dict_comb in self.parameters_combinations:

            self.attack = self.attack_class(**dict_comb)
            adversarial_samples, success_rate,perturbations = self.attack_dataloader(dataloader)

            #print(f"{[(key,value) for (key,value) in dict_comb.items() if key!='model']} accuracy: {success_rate}")
            filtered_comb = {k: v for k, v in dict_comb.items() if k != 'model'}
            self.comb_success[str(filtered_comb)] = [adversarial_samples, success_rate,perturbations]

        return self.comb_success

    def set_params(self, params):
        self.attack.set_params(**params)

    def parameters_permutation(self, params):
        keys, values = zip(*params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts


class SG_FAKEBOB(SG_Attack):

    def __init__(self):

        super(SG_FAKEBOB, self).__init__()
        self.attack_class = FAKEBOB  # Add this

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)

        adv_x, success = self.attack.attack(x=x_batch, y=y_batch)

        return adv_x,success


class SG_SIRENATTACK(SG_Attack):

    def __init__(self):

        super(SG_SIRENATTACK, self).__init__()
        self.attack_class = SirenAttack
        # Initialize FakeBOB specific parameters here

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        adv_x, success = self.attack.attack(x=x_batch, y=y_batch)

        return adv_x,success


class SG_FGSM(SG_Attack):

    def __init__(self):
        super(SG_FGSM, self).__init__()
        self.attack_class = FGSM  # Add this

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        adv_x,success = self.attack.attack_batch(x_batch,y_batch,lower = lower_batch,upper = upper_batch,batch_id = batch_id)
        return adv_x,success

class SG_PGD(SG_Attack):

    def __init__(self):
        super(SG_PGD, self).__init__()
        self.attack_class = PGD  # Add this

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        adv_x,success = self.attack.attack_batch(x_batch,y_batch,lower = lower_batch,upper = upper_batch,batch_id = batch_id)
        return adv_x,success

class SG_CW2(SG_Attack):

    def __init__(self):
        super(SG_CW2, self).__init__()
        self.attack_class = CW2  # Add this

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        adv_x,success = self.attack.attack_batch(x_batch,y_batch,lower = lower_batch,upper = upper_batch,batch_id = batch_id)
        return adv_x,success

class SG_CWInf(SG_Attack):

    def __init__(self):
        super(SG_CWInf, self).__init__()
        self.attack_class = CWinf  # Add this

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        adv_x,success = self.attack.attack_batch(x_batch,y_batch,lower = lower_batch,upper = upper_batch,batch_id = batch_id)
        return adv_x,success



class SG_KENAN(SG_Attack):

    def __init__(self):
        super(SG_KENAN, self).__init__()
        self.attack_class = Kenan  # Add this

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        adv_x,success = self.attack.attack_batch(x_batch=x_batch, y_batch=y_batch, batch_id=batch_id)
        return adv_x,success


class MultipleSGAttacks():
    def __init__(self,model,attacks,combinations,samples,labels):
        self.model = model
        self.attacks = attacks
        self.combinations = combinations
        self.samples = samples
        self.labels = labels
        self.dataset = CustomDataset(self.samples,self.labels)
        self.dataloader = DataLoader(self.dataset,batch_size = 1)
        self.results = {}
        self.results[self.model] = {}
        self.N_top = None

    def run_attacks(self, N_top,model = None):
        self.N_top = N_top
        if model is None:
            model = self.model


        for i, (attack_name, attack) in enumerate(self.attacks.items()):


            if attack_name in self.results[model].keys():
                print("Attack already done,skipping..")
                continue

            print(f"Running attack: {attack_name}")
            params = self.combinations[attack_name]
            combs = attack.attacks_combs(self.dataloader, params)
            attack_results = []
            for comb in combs.keys():
                adv_samples = combs[comb][0]

                ##
                samples_tensor = self.get_tensor(adv_samples)
                with torch.no_grad():
                    #print(samples_tensor.size())
                    predictions = model(samples_tensor)


                # Convert predictions to numpy array if necessary for further processing
                predictions = predictions.cpu().numpy()  # Remove '.cpu()' if not using CUDA
                predicted_labels = np.argmax(predictions, axis=1)  # predictions is already a numpy array
                true_labels = np.array([label.item() for label in self.labels])  # Convert list to numpy array
                accuracy = accuracy_score(true_labels, predicted_labels)
                print(f"{comb} has accuracy:{accuracy}")
                ##

                #success = combs[comb][1]

                success = accuracy

                perturbation = combs[comb][2]
                attack_results.append({
                    'params': comb,
                    'samples': adv_samples,
                    'success': success,
                    'perturbation': perturbation
                })

            # Sort results based on success rate in ascending order
            attack_results.sort(key=lambda x: x['success'])
            # Select top N_top results
            top_results = attack_results[:N_top]

            # Extract params and samples from top results
            top_params = [result['params'] for result in top_results]
            top_samples = [result['samples'] for result in top_results]

            # Store top results in self.results in the specified structure
            self.results[model][attack_name] = {
                'params': top_params,
                'samples': top_samples
            }



    def get_tensor(self,samples):
        # Convert samples to a PyTorch Tensor if they're not already
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(samples, torch.Tensor):
            samples_tensor = samples.type(torch.float32).to(device)

        # Check if samples is a list of tensors
        elif all(isinstance(sample, torch.Tensor) for sample in samples):
            samples_tensor = torch.stack(samples).type(torch.float32).to(device)
        # Check if samples is a list of numpy arrays
        elif all(isinstance(sample, np.ndarray) for sample in samples):
            samples_tensor = torch.tensor(np.stack(samples), dtype=torch.float32).to(device)
        # Check if samples is a single numpy array
        elif isinstance(samples, np.ndarray):
            samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
        else:
            raise ValueError("Unsupported sample type.")

        if samples_tensor.dim() == 4 and samples_tensor.size(2) == 1:
        # Assuming the input is [batch, channels, height, width]
        # and you want to treat 'width' as your sequence length
            samples_tensor = samples_tensor.squeeze()  # Remove singleton dimensions, result is [batch, seq_len]
            samples_tensor = samples_tensor.unsqueeze(1)  # Add feature dimension, result is [batch, seq_len, 1]
        elif samples_tensor.dim() != 3:
            raise ValueError(f"Unexpected tensor dimensions: {samples_tensor.dim()}")

        return samples_tensor

    def get_results(self):
        return self.results

    def set_results(self,results):
        self.results = results

    def get_mean_perturbation(self,samples_a,samples_b):
        individual_perturbations = []

        for sample_a, sample_b in zip(samples_a, samples_b):
            # Convert sample_or to a numpy array if it's a torch.Tensor
            if isinstance(sample_a, torch.Tensor):
                sample_a = sample_a.cpu().detach().numpy()

            # Ensure sample_adv is a numpy array (in case it's a torch.Tensor)
            if isinstance(sample_b, torch.Tensor):
                sample_b = sample_b.cpu().detach().numpy()

            # Compute the perturbation for the current pair
            perturbation = np.mean(np.abs(sample_b-sample_a))
            individual_perturbations.append(perturbation)

        return np.mean(individual_perturbations)

    def analyze_results(self,model_to_analyze = None):

        if model_to_analyze is None:
            model_to_analyze = self.model

        if model_to_analyze not in self.results.keys():
            self.results[model_to_analyze] = {}


        for attack_name, result in self.results[self.model].items():
            print(f"Results for {attack_name} on {str(model_to_analyze).split('(')[0]}:")
            top_params = result['params']
            top_samples = result['samples']
            #print(top_params)

            if attack_name not in self.results[model_to_analyze].keys():
                self.results[model_to_analyze][attack_name] = {}
                self.results[model_to_analyze][attack_name]['params'] = top_params
                self.results[model_to_analyze][attack_name]['samples'] = top_samples

            self.results[model_to_analyze][attack_name]['accuracies'] = []
            self.results[model_to_analyze][attack_name]['mean_perturbations'] = []
            self.results[model_to_analyze][attack_name]['mean_snrs'] = []

            for params, samples in zip(top_params, top_samples):

                samples_tensor = self.get_tensor(samples)

                # Ensure model is in evaluation mode
                model_to_analyze.eval()

                # Get predictions
                with torch.no_grad():
                    #print(samples_tensor.size())
                    predictions = model_to_analyze(samples_tensor)


                # Convert predictions to numpy array if necessary for further processing
                predictions = predictions.cpu().numpy()  # Remove '.cpu()' if not using CUDA
                predicted_labels = np.argmax(predictions, axis=1)  # predictions is already a numpy array
                true_labels = np.array([label.item() for label in self.labels])  # Convert list to numpy array

                accuracy = accuracy_score(true_labels, predicted_labels)
                snr_list = [SNR(sample,adv_sample) for adv_sample,sample in zip(samples,self.samples)]#self.calculate_snr(samples, self.samples)
                mean_snr = np.mean(snr_list)
                # Compute the overall mean perturbation
                mean_perturbation = self.get_mean_perturbation(samples,self.samples)

                print(f"Params: {params}, Accuracy: {accuracy}, Mean Perturbation: {mean_perturbation}, Mean snr: {mean_snr}")
                self.results[model_to_analyze][attack_name]['accuracies'].append(accuracy)
                self.results[model_to_analyze][attack_name]['mean_perturbations'].append(mean_perturbation)
                self.results[model_to_analyze][attack_name]['mean_snrs'].append(mean_snr)


    def plot_evaluation_curve(self,model = None):
        if model is None:
            model = self.model
        #print(self.results,self.results.keys())
        model_results = self.results[model]

        for attack_name, result in model_results.items():
            top_params = result['params']
            top_samples = result['samples']

            flattened_samples = top_samples[0].reshape(-1)  # Flatten into a single array
            max_v = np.max(flattened_samples)
            min_v = np.min(flattened_samples)
            max_pert = max_v - min_v

            accuracies = result['accuracies']
            mean_perturbations = result['mean_perturbations']

            # Pair, sort by mean perturbation in descending order, and unzip
            paired = list(zip(mean_perturbations, accuracies))
            paired.sort(reverse=True, key=lambda x: x[0])
            mean_perturbations_sorted, accuracies_sorted = zip(*paired)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(mean_perturbations_sorted, accuracies_sorted, label=f'{attack_name} Attack', marker='o')
            plt.xlabel('Mean Perturbation')
            plt.ylabel('Accuracy')
            plt.title(f'Evaluation Curve for {attack_name} Attack')
            plt.legend()
            plt.grid(True)
            #plt.tight_layout()
            # Set x-axis and y-axis limits
            plt.xlim(0, max_pert)
            margin = 0.1
            plt.ylim(0 - margin, 1 + margin)

            plt.show()


    def plot_evaluation_curve_snr(self,model = None):
        if model is None:
            model = self.model
        model_results = self.results[model]
        for attack_name, result in model_results.items():
            top_params = result['params']
            top_samples = result['samples']

            accuracies = result['accuracies']
            mean_snrs = result['mean_snrs']

            # Pair, sort by mean SNR in descending order, and unzip
            paired = list(zip(mean_snrs, accuracies))
            paired.sort(reverse=True, key=lambda x: x[0])
            mean_snrs_sorted, accuracies_sorted = zip(*paired)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(mean_snrs_sorted, accuracies_sorted, label=f'{attack_name} Attack', marker='o')
            plt.xlabel('Mean Signal-to-Noise Ratio (SNR)')
            plt.ylabel('Accuracy')
            plt.title(f'Evaluation Curve for {attack_name} Attack (Mean SNR vs. Accuracy)')
            plt.legend()
            plt.grid(True)

            mean_snrs = np.array(mean_snrs)  # Ensure it's a numpy array
            mean_snrs = mean_snrs[np.isfinite(mean_snrs)]  # Keep only finite values

            plt.xlim(-30,0)
            margin = 0.1
            plt.ylim(0-margin, 1+margin)
            plt.show()



    def plot_top_n_attacks(self, N,model = None):
        """
        For each attack, plot a separate chart where the x-axis shows the parameter combinations
        (e.g., {"eps = 5, eps_step = 1"}, {"eps = 3, eps_step = 0.5"}, ...) and the y-axis shows
        the accuracies of those combinations (e.g., 0.97, 0.95, ...).
        """
        if model is None:
            model = self.model
        model_results = self.results[model]
        all_results = {}
        for attack_name, result in model_results.items():
            top_params = result['params']
            top_samples = result['samples']
            accuracies = result['accuracies']

            attack_results = [(params,accuracy) for params,accuracy in zip(top_params,accuracies)]
            '''
            for params, samples in zip(top_params, top_samples):
                samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(samples_tensor)
                predictions = predictions.cpu().numpy()
                predicted_labels = np.argmax(predictions, axis=1)
                accuracy = np.mean(predicted_labels == self.labels)
                accuracy = round(accuracy,3)


                accuracies.append(accuracy)
                attack_results.append((params, accuracy))
            '''
            print(attack_results)
            # Sort results for each attack based on accuracy
            sorted_results = sorted(attack_results, key=lambda x: x[1])[:N]
            print(sorted_results)
            all_results[attack_name] = sorted_results

        # Plotting
        for attack_name, sorted_results in all_results.items():
            #param_combinations = [f"{', '.join(f'{k}={v}' for k, v in params.items())}" for params, _ in sorted_results]
            param_combinations = []
            for params, _ in sorted_results:
                if isinstance(params, dict):
                    param_string = ', '.join(f'{k}={v}' for k, v in params.items())
                elif isinstance(params, str):
                    param_string = params  # if params is a string, use it directly
                else:
                    param_string = str(params)  # for other types, convert to string
                param_combinations.append(param_string)

            accuracies = [accuracy for _, accuracy in sorted_results]

            plt.figure(figsize=(10, 5))
            plt.plot(param_combinations, accuracies, color='skyblue')
            plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
            plt.xlabel('Parameter Combinations')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy of Parameter Combinations for {attack_name} Attack')
            #plt.tight_layout()  # Adjust layout to fit rotated x-axis labels
            margin = 0.1
            plt.ylim(0 - margin, 1 + margin)
            plt.grid(True)
            plt.show()


    def get_effective_samples(self, attacks, top_attacks = None, samples_per_attack = None):

        device = next(self.model.parameters()).device  # Get the device the model is on
        adv_samples_per_attack = {}
        samples_params = {}
        original_samples_tensor = self.get_tensor(self.samples)

        if top_attacks is None:
            top_attacks = self.N_top

        for attack_name, attack_dict in self.results[self.model].items():
            if attack_name not in attacks:
                continue
            adv_samples_per_attack[attack_name] = []

            params = attack_dict['params']
            list_adv_samples = attack_dict['samples'][:top_attacks]

            for i, adv_samples in enumerate(list_adv_samples):
                # Check shapes and types of adv_samples before converting to tensor

                adv_samples_tensor = self.get_tensor(adv_samples)

                self.model.eval()

                with torch.no_grad():
                    original_predictions = self.model(original_samples_tensor).argmax(dim=1)
                    adversarial_predictions = self.model(adv_samples_tensor).argmax(dim=1)
                    #print(original_predictions,adversarial_predictions)
                    # Find samples where predictions differ and original prediction is zero
                    diff_indices = torch.where((original_predictions != adversarial_predictions) & (original_predictions == 0))[0]
                    effective_advs = adv_samples_tensor[diff_indices]
                    added_samples = 0
                    # Add effective adversarial samples to the list, up to the desired number
                    for adv in effective_advs:
                        adv_samples_per_attack[attack_name].append(adv.cpu().numpy())  # Convert back to numpy if needed
                        added_samples += 1
                        samples_params[str(adv.cpu().numpy())] = params
                        if samples_per_attack is not None and added_samples >= samples_per_attack:
                            break

        return adv_samples_per_attack,samples_params


    def get_random_samples(self, N):
        random_samples_per_attack = {}

        # Ensure each attack contributes equally to the total N samples
        samples_per_attack = N // len(self.results)

        for attack_name, attack_dict in self.results[self.model].items():
            # Get all samples for the current attack
            all_samples = attack_dict['samples']
            flat_samples = [item for sublist in all_samples for item in sublist]  # Flatten the list of lists

            # Randomly select samples from the flattened list
            selected_samples = random.sample(flat_samples, min(samples_per_attack, len(flat_samples)))

            # If the samples are tensors, convert them to numpy arrays
            selected_samples = [sample.cpu().numpy() if isinstance(sample, torch.Tensor) else sample for sample in selected_samples]

            random_samples_per_attack[attack_name] = selected_samples

        return random_samples_per_attack


    def show_adversarial_samples(self, attacks, top_attacks, samples_per_attack):
        model = self.model
        device = next(model.parameters()).device  # Get the device the model is on
        model.eval()  # Set the model to evaluation mode
        sample_rate = 16000

        effective_samples,samples_params = self.get_effective_samples(attacks, top_attacks, samples_per_attack)
        original_samples_tensor = self.get_tensor(self.samples)

        for attack_name, adv_samples in effective_samples.items():
            #print(len(adv_samples))
            if len(adv_samples)==0:
                continue
            adv_samples_tensor = self.get_tensor(adv_samples)

            with torch.no_grad():
                original_predictions = self.model(original_samples_tensor).argmax(dim=1)
                #print(self.model(adv_samples_tensor))
                adversarial_predictions = self.model(adv_samples_tensor).argmax(dim=-1)#try -1

            for i, (original_prediction, adversarial_prediction) in enumerate(zip(original_predictions, adversarial_predictions)):

                if original_prediction == 1 or original_prediction == adversarial_prediction:
                    continue
                params = samples_params[str(adv_samples[i])]
                original_sample = self.samples[i].squeeze()
                adv_sample = adv_samples[i].squeeze()
                print(f"Attack :{attack_name} params: {params}")
                compare_samples(original_sample,adv_sample,sample_rate, title1=f"Original sample (Prediction {original_prediction.item()})", title2=f"{attack_name} Adversarial sample (Prediction {adversarial_prediction.item()})")

    def compare_models(self, models=None, models_name={}):
        model_accuracies = {}
        model_params = {}

        if models is None:
            models = self.results.keys()

        # Iterate through each model and their results
        for model_name, model_results in self.results.items():
            if model_name not in models:
                continue

            if model_name not in model_accuracies:
                model_accuracies[model_name] = {}
                model_params[model_name] = {}

            for attack_name, result in model_results.items():
                # Store accuracies and a string representation of parameters for each attack
                model_accuracies[model_name][attack_name] = result['accuracies']
                model_params[model_name][attack_name] = result['params']

        # Determine the number of attacks to plot
        num_attacks = len(set(k for d in model_accuracies.values() for k in d))
        fig, axes = plt.subplots(1, num_attacks, figsize=(14 * num_attacks, 8))  # Adjust the total width as needed

        # Ensure axes is an array even with a single subplot
        if num_attacks == 1:
            axes = [axes]

        # Plotting for each attack
        for ax, attack_name in zip(axes, sorted(set(k for d in model_accuracies.values() for k in d))):
            colors = plt.cm.jet(np.linspace(0, 1, len(model_accuracies)))
            line_styles = ['-', '--', '-.', ':']  # Different line styles for differentiation
            marker_types = ['o', 's', '^', 'p', '*']  # Different markers for further differentiation

            for model_idx, (model_name, attacks) in enumerate(model_accuracies.items()):
                if attack_name in attacks:
                    accs = attacks[attack_name]
                    params = model_params[model_name][attack_name]
                    if model_name not in models_name.keys():
                        label = str(model_name).split("(")[0]
                    else:
                        label = models_name[model_name]

                    # Choose line style and marker based on model index
                    line_style = line_styles[model_idx % len(line_styles)]
                    marker_type = marker_types[model_idx % len(marker_types)]

                    ax.plot(params, accs, label=label, marker=marker_type, linestyle=line_style, color=colors[model_idx])

            ax.set_xticklabels(params, rotation=90, ha="right", fontsize='x-small')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Parameters')
            ax.set_title(f'Accuracy Comparison for {attack_name}')
            ax.legend()
            ax.grid(True)
            margin = 0.1
            ax.set_ylim(0 - margin, 1 + margin)

        plt.tight_layout()
        plt.show()
