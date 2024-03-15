#@title Models classes

import numpy as np
import torch
from torch import nn, optim
import torchaudio
import torch.nn.functional as F



class FullyConnected(torch.nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self, n_feature: int, n_hidden: int, dropout: float, relu_max_clip: int = 20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x

class DeepSpeech(torch.nn.Module):
    """DeepSpeech architecture introduced in
    *Deep Speech: Scaling up end-to-end speech recognition* :cite:`hannun2014deep`.

    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
        n_class: Number of output classes
    """

    def __init__(
        self,
        n_feature: int,
        n_hidden: int = 2048,
        n_class: int = 40,
        dropout: float = 0.0,
    ) -> None:
        super(DeepSpeech, self).__init__()
        self.n_hidden = n_hidden
        self.fc1 = FullyConnected(n_feature, n_hidden, dropout)
        self.fc2 = FullyConnected(n_hidden, n_hidden, dropout)
        self.fc3 = FullyConnected(n_hidden, n_hidden, dropout)
        self.bi_rnn = torch.nn.RNN(n_hidden, n_hidden, num_layers=1, nonlinearity="relu", bidirectional=True)
        self.fc4 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = torch.nn.Linear(n_hidden, n_class)

        self.decision_threshold = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
        Returns:
            Tensor: Predictor tensor of dimension (batch, time, class).
        """
        # Ensure x has a batch dimension even if batch_size = 1
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        #x = x.view(x.size(0),1,x.size(2)*x.size(3))#Test

        # x is initially [batch size, 1, length], we want [1, batch size, length] for RNN
        #print(x.size())
        x = x.squeeze(1)  # Remove the channel dimension, resulting in [batch size, length]
        #print(x.size())
        x = x.unsqueeze(0)  # Add a time dimension, resulting in [1, batch size, length]
        #print(x.size())



        # N x C x T x F
        x = self.fc1(x)
        # N x C x T x H
        x = self.fc2(x)
        # N x C x T x H
        x = self.fc3(x)
        # N x C x T x H
        #print(x.shape)
        #x = x.squeeze(1)  # Ensure this operation is producing the expected shape.
        #print(x.shape)
        x = x.transpose(0, 1)  # Transposing to get T x N x H
        #print(x.shape)
        x, _ = self.bi_rnn(x)  # Output should be T x N x 2*H for bidirectional RNN
        #print(x.shape)
        # The fifth (non-recurrent) layer takes both the forward and backward units as inputs
        x = x[:, :, : self.n_hidden] + x[:, :, self.n_hidden :]
        # T x N x H
        x = self.fc4(x)
        # T x N x H
        x = self.out(x)
        # T x N x n_class
        x = x.permute(1, 0, 2)
        # N x T x n_class
        x = torch.nn.functional.log_softmax(x, dim=2)
        #x = torch.nn.functional.softmax(x, dim=2)
        # N x T x n_class

        #added 0 to handle batch_size = 1
        x = x.squeeze(0)


        return x


    def score(self, x, softmax=True):
        x = self.forward(x)
        # Ensure x has at least 2 dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if softmax:
            # Apply softmax on the last dimension
            x = F.softmax(x, dim=-1)
        return x

    def get_threshold(self):
        return self.decision_threshold

    def set_threshold(self, threshold: float) -> None:
        """Sets the decision threshold for the model.

        Args:
            threshold: The new threshold value.
        """
        self.decision_threshold = threshold

    def make_decision(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        scores = self.score(x, softmax=True)

        # Apply thresholding
        #decisions = scores >= self.decision_threshold
        decisions = torch.argmax(scores, dim=-1)

        return decisions, scores


    def score(self, x, softmax=True):
        x = self.forward(x)
        # Ensure x has at least 2 dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if softmax:
            # Apply softmax on the last dimension
            x = F.softmax(x, dim=-1)
        return x


    def make_decision(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Before feeding `x` to the model
        x = x.to(device)
        scores = self.score(x, softmax=True)
        # Ensure scores has at least 2 dimensions
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        decisions = torch.argmax(scores, dim=-1)  # Apply argmax on the last dimension
        return decisions, scores



class CombinedDetectorModel(nn.Module):
    def __init__(self, detector, main_model):
        super(CombinedDetectorModel, self).__init__()
        self.detector = detector
        self.main_model = main_model

    def forward(self, x):
        # Pass the input through the detector
        detector_outputs = self.detector(x)
        detector_predicted_labels = torch.argmax(detector_outputs, dim=1)

        # Initialize a tensor to hold the output
        final_outputs = torch.zeros_like(detector_outputs)

        # For inputs where the detector predicts 1 (clean), pass them through the main model
        clean_mask = (detector_predicted_labels == 1)
        if clean_mask.any():
            clean_inputs = x[clean_mask]
            clean_outputs = self.main_model(clean_inputs)
            final_outputs[clean_mask] = clean_outputs

        # For inputs where the detector predicts 0 (corrupted), the output remains zero (or you can set it to any specific value)
        # Note: This step is optional as final_outputs are initialized to zeros

        return final_outputs


class CombinedDefenseModel(nn.Module):
    def __init__(self, defenses, main_model):
        super(CombinedDefenseModel, self).__init__()
        self.defenses = defenses
        self.main_model = main_model

    def forward(self, x):
        # Pass the input through the detector
        for defense in self.defenses:
            x = defense(x)
        #x = normalize_audio(x)


        x = self.main_model(x)
        #print(x)
        return x





import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        #input = input.unsqueeze(1)
        #input = F.pad(input, (1, 0), 'reflect')
        return torch.nn.functional.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C,num_classes):

        super(ECAPA_TDNN, self).__init__()
        self.num_classes = num_classes
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        #added
        self.fc7 = nn.Linear(192, self.num_classes)



    def forward(self, x,aug=True):

        # Ensure x has a batch dimension even if batch_size = 1
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        x = self.fc7(x)

        x = torch.nn.functional.log_softmax(x, dim=-1)
        #print(x)
        #x = x.squeeze(0)
        #print(x)



        return x
