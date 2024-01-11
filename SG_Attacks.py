import itertools
import torch
import numpy as np
from tqdm import tqdm

from SpeakerGuard.attack.FAKEBOB import FAKEBOB
from SpeakerGuard.attack.FGSM import FGSM
from SpeakerGuard.attack.PGD import PGD
from SpeakerGuard.attack.CW2 import CW2
from SpeakerGuard.attack.CWinf import CWinf
from SpeakerGuard.attack.Kenan import Kenan
from SpeakerGuard.attack.SirenAttack import SirenAttack

class SG_Attack:
    def __init__(self):
        pass

    def attack(self, x, y):
        raise NotImplementedError

    def attack_batch(self, x_batch, y_batch, **kwargs):
        raise NotImplementedError

    
class SG_FAKEBOB(SG_Attack):
    def __init__(self):
        super(SG_FAKEBOB, self).__init__()
        
        # Initialize FakeBOB specific parameters here

    def attack_dataloader(self, dataloader):
        success_counter = 0
        adversarial_samples = []

        # tqdm is used here to wrap the dataloader, providing a progress bar
        for i, (x_batch, y_batch) in enumerate(tqdm(dataloader, desc="Attacking")):
            print(i)
            # Perform the attack on the current batch
            adv_x, success = self.attack_batch(x_batch, y_batch, batch_id=i)
            print("*"*10)
            # Accumulate the number of successful attacks
            success_counter += success

            # Store the adversarial examples
            adversarial_samples.append(adv_x)

        # Calculate the success rate of the attack
        success_rate = success_counter / (i + 1)

        return adversarial_samples, success_rate
    

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        print("at..")
        adv_x,success = self.attack.attack_batch(x_batch,y_batch,lower = lower,upper = upper,batch_id = batch_id)
        return adv_x,success
    

    def preprocess_batch(self,x_batch,y_batch):
        #while len(x_batch.size())>2:
        #    x_batch = x_batch.squeeze(0)
        x_reshaped = x_batch.clone().detach().unsqueeze(0)  
        x_min = x_reshaped.min()
        x_max = x_reshaped.max()
        x_reshaped_normalized = 2 * ((x_reshaped - x_min) / (x_max - x_min)) - 1
        return x_reshaped_normalized,y_batch

    def attacks_combs(self,dataloader,params):

        self.parameters_combinations = self.parameters_permutation(params)
        self.comb_success = {}

        for dict_comb in self.parameters_combinations:
            self.attack = FAKEBOB(**dict_comb)
            adversarial_samples,success_rate = self.attack_dataloader(dataloader)
            self.comb_success[dict_comb] = (adversarial_samples,success_rate)
        
        return self.comb_success
    
    def set_params(self,params):
        self.attack.set_params(**params)
        #self.attack.set_params(**{'verbose': False})

    def parameters_permutation(self,params):
        keys, values = zip(*params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts
    
class SG_FGSM(SG_Attack):
    def __init__(self):
        super(SG_FGSM, self).__init__()
        
        # Initialize FakeBOB specific parameters here

    def attack_dataloader(self, dataloader):
        success_counter = 0
        adversarial_samples = []

        # tqdm is used here to wrap the dataloader, providing a progress bar
        for i, (x_batch, y_batch) in enumerate(tqdm(dataloader, desc="Attacking")):
            # Perform the attack on the current batch
            adv_x, success = self.attack_batch(x_batch, y_batch, batch_id=i)

            # Accumulate the number of successful attacks
            success_counter += success

            # Store the adversarial examples
            adversarial_samples.append(adv_x)

        # Calculate the success rate of the attack
        success_rate = success_counter / (i + 1)

        return adversarial_samples, success_rate
    

    def attack_batch(self, x_batch, y_batch, batch_id,**kwargs):

        x_batch,y_batch = self.preprocess_batch(x_batch,y_batch)

        lower = -1
        upper = 1
        lower_batch = torch.full_like(x_batch, lower)
        upper_batch = torch.full_like(x_batch, upper)
        print(x_batch.size(),x_batch)
        adv_x,success = self.attack.attack_batch(x_batch,y_batch,lower = lower,upper = upper,batch_id = batch_id)
        return adv_x,success
    

    def preprocess_batch(self,x_batch,y_batch):
        #while len(x_batch.size())>2:
        #    x_batch = x_batch.squeeze(0)
        x_reshaped = x_batch.clone().detach().unsqueeze(0)  
        x_min = x_reshaped.min()
        x_max = x_reshaped.max()
        x_reshaped_normalized = 2 * ((x_reshaped - x_min) / (x_max - x_min)) - 1
        return x_reshaped_normalized,y_batch

    def attacks_combs(self,dataloader,params):

        self.parameters_combinations = self.parameters_permutation(params)
        self.comb_success = {}

        for dict_comb in self.parameters_combinations:
            self.attack = FGSM(**dict_comb)
            adversarial_samples,success_rate = self.attack_dataloader(dataloader)
            self.comb_success[dict_comb] = (adversarial_samples,success_rate)
        
        return self.comb_success
    
    def set_params(self,params):
        self.attack.set_params(**params)
        #self.attack.set_params(**{'verbose': False})

    def parameters_permutation(self,params):
        keys, values = zip(*params.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_dicts


