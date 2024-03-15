#@title Defenses classes

import torch

import torchaudio
from scipy import signal
import torch_lfilter
from torch_lfilter import lfilter
from torch.utils.data import ConcatDataset, random_split
import inspect
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import os
from scipy.io.wavfile import read, write
import shlex
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

'''
Input_Transformation = [
    'QT', 'AT', 'AS', 'MS', # Time Domain
    'DS', 'LPF', 'BPF', # Frequency Domain
    'OPUS', 'SPEEX', 'AMR', 'AAC_V', 'AAC_C', 'MP3_V', 'MP3_C', # Speech Compression
    'FEATURE_COMPRESSION', # Feature-Level,; Ours
    'FeCo', # Feature-Level,; Ours; abbr,
]
'''
def normalize_audio(audio, **kwargs):
    """
    Normalizes the audio signal to the -1 to 1 range based on peak amplitude.

    Parameters:
    - audio: A PyTorch tensor representing the audio samples.
    - **kwargs: Ignored parameters, included for compatibility with the defense framework.

    Returns:
    - A PyTorch tensor of the normalized audio.
    """
    assert torch.is_tensor(audio), "Input must be a PyTorch tensor"

    if audio.numel() == 0:  # Check if the tensor is empty
        raise ValueError("Audio tensor is empty.")

    peak_amplitude = torch.max(torch.abs(audio))
    if peak_amplitude > 0:
        normalized_audio = audio / peak_amplitude
    else:
        # This branch might be unnecessary for non-silent audio,
        # but it's kept here for completeness and safety.
        normalized_audio = audio.clone()  # Avoid in-place modification

    return normalized_audio

def normalize_audio_batch(audio_batch):
    """
    Normalizes each audio signal in a batch to the -1 to 1 range based on its peak amplitude.

    Parameters:
    - audio_batch: A PyTorch tensor representing the batch of audio samples with shape (batch_size, num_samples).

    Returns:
    - A PyTorch tensor of the normalized audio batch.
    """
    assert torch.is_tensor(audio_batch), "Input must be a PyTorch tensor"
    if audio_batch.numel() == 0:  # Check if the tensor is empty
        raise ValueError("Audio tensor is empty.")

    audio_batch = audio_batch.squeeze()

    # Ensure it's a batched input
    if audio_batch.ndim != 2:
        raise ValueError(f"Input tensor must be 2-dimensional (batch_size, num_samples) found {audio_batch.shape}.")

    peak_amplitudes = torch.max(torch.abs(audio_batch), dim=1, keepdim=True).values

    # Avoid division by zero for silent audio (all zeros)
    peak_amplitudes[peak_amplitudes == 0] = 1

    normalized_audio_batch = audio_batch / peak_amplitudes

    normalized_audio_batch = normalized_audio_batch.unsqueeze(1)

    return normalized_audio_batch

def standardize_audio(audio, **kwargs):
    """
    Standardizes the audio signal to have a mean of 0 and a standard deviation of 1.

    Parameters:
    - audio: A PyTorch tensor representing the audio samples.
    - **kwargs: Ignored parameters, included for compatibility with the defense framework.

    Returns:
    - A PyTorch tensor of the standardized audio.
    """
    assert torch.is_tensor(audio), "Input must be a PyTorch tensor"

    if audio.numel() == 0:  # Check if the tensor is empty
        raise ValueError("Audio tensor is empty.")



    # Calculate the mean and standard deviation of the audio tensor
    mean = torch.mean(audio)
    std = torch.std(audio)

    # Handle case where standard deviation is zero (all values are the same)
    if std > 0:
        standardized_audio = (audio - mean) / std
    else:
        # When std is 0, the audio signal is constant; standardization is not applicable.
        # One approach is to return a tensor of zeros of the same shape.
        standardized_audio = torch.zeros_like(audio)  # This creates a tensor of zeros with the same shape as audio

    return standardized_audio


def standardize_audio_batch(audio_batch):
    """
    Standardizes each audio signal in a batch to have a mean of 0 and a standard deviation of 1.

    Parameters:
    - audio_batch: A PyTorch tensor representing the batch of audio samples with shape (batch_size, num_samples).

    Returns:
    - A PyTorch tensor of the standardized audio batch.
    """
    assert torch.is_tensor(audio_batch), "Input must be a PyTorch tensor"

    if audio_batch.numel() == 0:  # Check if the tensor is empty
        raise ValueError("Audio tensor is empty.")

    audio_batch = audio_batch.squeeze()

    # Ensure it's a batched input
    if audio_batch.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional (batch_size, num_samples).")

    # Calculate the mean and standard deviation for each audio signal in the batch
    means = torch.mean(audio_batch, dim=1, keepdim=True)
    stds = torch.std(audio_batch, dim=1, keepdim=True)

    # Prevent division by zero by setting stds of 0 to 1 (affects constant audio signals)
    stds[stds == 0] = 1

    # Standardize each audio signal in the batch
    standardized_audio_batch = (audio_batch - means) / stds

    standardized_audio_batch = standardized_audio_batch.unsqueeze(1)
    return standardized_audio_batch

def convert_format_batch(audio_batch, format, back=False):
    """
    Converts a batch of audio signals to the specified format and optionally back to WAV.

    Args:
        audio_batch (torch.Tensor): Input batch of audio samples (batch_size, 1, num_samples).
        format (str): Target format ('mp3', 'ogg', etc.).
        back (bool, optional): If True, converts the audio back to WAV after the initial conversion. Defaults to False.

    Returns:
        torch.Tensor: Audio data after conversion(s), same shape as input.
    """
    assert torch.is_tensor(audio_batch), "Input must be a PyTorch tensor"
    assert audio_batch.ndim == 3, "Input tensor must be 3-dimensional (batch_size, 1, num_samples)"

    new_audio_batch = []
    audio_batch = audio_batch.cpu()  # Ensure on CPU

    for audio_signal in audio_batch:
        with NamedTemporaryFile(suffix='.wav') as tmp_wav, \
             NamedTemporaryFile(suffix='.' + format) as tmp_converted:

            # Save original WAV (clone & detach to avoid computation graph issues)
            audio_signal_clone = audio_signal.clone().detach()
            torchaudio.save(tmp_wav.name, audio_signal_clone, 16000)

            # Convert to target format
            audio = AudioSegment.from_wav(tmp_wav.name)
            audio.export(tmp_converted.name, format=format)

            if back:
                # Convert back to WAV
                with NamedTemporaryFile(suffix='.wav') as tmp_wav_back:
                    audio = AudioSegment.from_file(tmp_converted.name, format=format)  # Load in 'format'
                    audio.export(tmp_wav_back.name, format='wav')

                    # Load WAV as tensor
                    audio, _ = torchaudio.load(tmp_wav_back.name, normalize=True)
            else:
                # Load the converted format as tensor
                audio, _ = torchaudio.load(tmp_converted.name, normalize=True)

            new_audio_batch.append(audio)

    recovered_audio_batch = torch.cat(new_audio_batch, dim=0).unsqueeze(1)
    return recovered_audio_batch



def batch_low_pass_filter(audio, cutoff_freq, sample_rate):
    """
    Applies a low-pass filter to a batch of 1D audio signals using FFT in the frequency domain.

    Parameters:
    - audio: A 3D PyTorch tensor representing the batch of audio samples with shape [batch_size, 1, sample_length].
    - cutoff_freq: The cutoff frequency for the low-pass filter.
    - sample_rate: The sample rate of the audio signals.

    Returns:
    - A 3D PyTorch tensor of the filtered audio with the same shape as the input.
    """
    # Ensure the input audio is a 3D tensor
    assert audio.dim() == 3, "Input audio must be a 3D tensor [batch_size, 1, sample_length]."

    # FFT operates on the last dimension, so we reshape audio for batch processing
    # Reshape to [batch_size * 1, sample_length] for batch FFT
    batch_size, channels, sample_length = audio.shape
    audio_reshaped = audio.reshape(-1, sample_length)

    # Perform FFT on the last dimension
    audio_fft = torch.fft.rfft(audio_reshaped)

    # Calculate frequencies
    freqs = torch.fft.rfftfreq(sample_length, d=1/sample_rate)

    # Create a low-pass filter mask for frequencies above the cutoff
    mask = torch.abs(freqs) <= cutoff_freq

    # Apply the mask: Use broadcasting to apply it across the batch
    filtered_fft = audio_fft * mask.unsqueeze(0).to(audio_fft.device)  # Use .device attribute to match devices

    # Perform the inverse FFT
    filtered_audio = torch.fft.irfft(filtered_fft, n=sample_length)

    # Reshape back to the original audio tensor shape
    filtered_audio_reshaped = filtered_audio.reshape(batch_size, channels, sample_length)

    return filtered_audio_reshaped

def batch_high_pass_filter(audio, cutoff_freq, sample_rate, **kwargs):
    """
    Applies a low-pass filter to a batch of 1D audio signals using FFT in the frequency domain.

    Parameters:
    - audio: A 3D PyTorch tensor representing the batch of audio samples with shape [batch_size, 1, sample_length].
    - cutoff_freq: The cutoff frequency for the low-pass filter.
    - sample_rate: The sample rate of the audio signals.

    Returns:
    - A 3D PyTorch tensor of the filtered audio with the same shape as the input.
    """
    # Ensure the input audio is a 3D tensor
    assert audio.dim() == 3, "Input audio must be a 3D tensor [batch_size, 1, sample_length]."

    # FFT operates on the last dimension, so we reshape audio for batch processing
    # Reshape to [batch_size * 1, sample_length] for batch FFT
    batch_size, channels, sample_length = audio.shape
    audio_reshaped = audio.reshape(-1, sample_length)

    # Perform FFT on the last dimension
    audio_fft = torch.fft.rfft(audio_reshaped)

    # Calculate frequencies
    freqs = torch.fft.rfftfreq(sample_length, d=1/sample_rate)

    # Create a high-pass filter mask for frequencies above the cutoff
    mask = torch.abs(freqs) > cutoff_freq

    # Apply the mask: Use broadcasting to apply it across the batch
    filtered_fft = audio_fft * mask.unsqueeze(0).to(audio_fft.device)  # Use .device attribute to match devices

    # Perform the inverse FFT
    filtered_audio = torch.fft.irfft(filtered_fft, n=sample_length)

    # Reshape back to the original audio tensor shape
    filtered_audio_reshaped = filtered_audio.reshape(batch_size, channels, sample_length)

    return filtered_audio_reshaped

def low_pass_filter(audio, cutoff_freq, sample_rate):
    """
    Applies a low-pass filter to a 1D audio signal using FFT in the frequency domain.

    Parameters:
    - audio: A 2D PyTorch tensor representing the audio sample with shape [1, sample_length].
    - cutoff_freq: The cutoff frequency for the low-pass filter.
    - sample_rate: The sample rate of the audio signal.

    Returns:
    - A 2D PyTorch tensor of the filtered audio with the same shape as the input.
    """
    #print(audio.shape)
    if audio.shape[0]==1:
        audio = audio.squeeze(0)
    #print(audio.shape)
    # Perform FFT on the audio
    audio_fft = torch.fft.rfft(audio)

    # Calculate frequencies
    sample_length = audio.shape[1]
    freqs = torch.fft.rfftfreq(sample_length, d=1/sample_rate)

    # Create a low-pass filter mask for frequencies above the cutoff
    mask = torch.abs(freqs) <= cutoff_freq

    # Apply the mask
    filtered_fft = audio_fft * mask.to(audio_fft.device)  # Match devices

    # Perform the inverse FFT
    filtered_audio = torch.fft.irfft(filtered_fft, n=sample_length)

    return filtered_audio

def high_pass_filter(audio, cutoff_freq, sample_rate):
    """
    Applies a high-pass filter to a 1D audio signal using FFT in the frequency domain.

    Parameters:
    - audio: A 2D PyTorch tensor representing the audio sample with shape [1, sample_length].
    - cutoff_freq: The cutoff frequency for the high-pass filter.
    - sample_rate: The sample rate of the audio signal.

    Returns:
    - A 2D PyTorch tensor of the filtered audio with the same shape as the input.
    """
    if audio.shape[0]==1:
        audio = audio.squeeze(0)
    # Perform FFT on the audio
    audio_fft = torch.fft.rfft(audio)

    # Calculate frequencies
    sample_length = audio.shape[1]
    freqs = torch.fft.rfftfreq(sample_length, d=1/sample_rate)

    # Create a high-pass filter mask for frequencies below the cutoff
    mask = torch.abs(freqs) > cutoff_freq

    # Apply the mask
    filtered_fft = audio_fft * mask.to(audio_fft.device)  # Match devices

    # Perform the inverse FFT
    filtered_audio = torch.fft.irfft(filtered_fft, n=sample_length)

    return filtered_audio

class BPDA(nn.Module):

    def __init__(self, ori_f, sub_f):
        """[summary]

        Parameters
        ----------
        ori_f : [type]
            Currently BPDA not supports ori_f with * and ** parameter;
            Only support ori_f like defense/time_domain/QT_Non_Diff
        sub_f : [type]
            Should accept the same number of input and return the same number of outputs as ori_f
        """
        super().__init__()
        self.f = self.get_diff_func(ori_f, sub_f)
        ori_f_args = inspect.getfullargspec(ori_f).args
        self.ori_f_defaults = inspect.getfullargspec(ori_f).defaults # maybe None --> no default parameters
        self.ori_f_num_required = len(ori_f_args) - len(self.ori_f_defaults) if self.ori_f_defaults else len(ori_f_args)
        self.ori_f_option_parameters = ori_f_args[-len(self.ori_f_defaults):] if self.ori_f_defaults else []

    def get_diff_func(self, ori_f, sub_f):

        class differ_func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                ctx.args = args
                return ori_f(*args)

            @staticmethod
            @torch.enable_grad()
            def backward(ctx, *grad_output):
                inputs = ctx.args
                inputs_all = []
                inputs_need_grad = []
                for input_ in inputs:
                    if torch.is_tensor(input_): # TO DO: change to float or double tensor
                        input_ = input_.detach().clone().requires_grad_()
                        inputs_need_grad.append(input_)
                    inputs_all.append(input_)
                outputs = sub_f(*inputs_all)
                num_output_ori = len(grad_output)
                num_output_sub = len(outputs) if isinstance(outputs, (tuple, list)) else 1
                assert num_output_ori == num_output_sub, 'The number of outputs of sub_f mismatches with ori_f'
                return torch.autograd.grad(outputs, inputs_need_grad, *grad_output) + tuple([None] * (len(inputs_all) - len(inputs_need_grad)))

        return differ_func

    def forward(self, *args, **kwargs):
        if len(list(kwargs.keys())) > 0:
            args = list(args)
            start = len(args) - self.ori_f_num_required
            for k, v in zip(self.ori_f_option_parameters[start:], self.ori_f_defaults[start:]):
                if k in kwargs.keys():
                    args.append(kwargs[k])
                else:
                    args.append(v)
            args = tuple(args)
        return self.f.apply(*args)



class SpeakerGuardDefenses():

    def __init__(self):
        pass

    def DS(audio, param=0.5, fs=16000, same_size=True):

        assert torch.is_tensor(audio) == True
        assert torch.is_tensor(audio) == True
        ori_shape = audio.shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0) # (T, ) --> (1, T)
        elif len(audio.shape) == 2: # (B, T)
            pass
        elif len(audio.shape) == 3:
            audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        down_ratio = param
        new_freq = int(fs * down_ratio)
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_freq, resampling_method='sinc_interpolation').to(audio.device)
        up_sampler = torchaudio.transforms.Resample(orig_freq=new_freq, new_freq=fs, resampling_method='sinc_interpolation').to(audio.device)
        down_audio = resampler(audio)
        new_audio = up_sampler(down_audio)
        if same_size: ## sometimes the returned audio may have longer size (usually 1 point)
            return new_audio[..., :audio.shape[1]].view(ori_shape)
        else:
            return new_audio.view(ori_shape[:-1] + new_audio.shape[-1:])

    def LPF(new, fs=16000, wp=4000, param=8000, gpass=3, gstop=40, same_size=True, bits=16):

        assert torch.is_tensor(new) == True
        ori_shape = new.shape
        if len(new.shape) == 1:
            new = new.unsqueeze(0) # (T, ) --> (1, T)
        elif len(new.shape) == 2: # (B, T)
            pass
        elif len(new.shape) == 3:
            new = new.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        if 0.9 * new.max() <= 1 and 0.9 * new.min() >= -1:
            clip_max = 1
            clip_min = -1
        else:
            clip_max = 2 ** (bits - 1) - 1
            clip_min = -2 ** (bits - 1)

        ws = param
        wp = 2 * wp / fs
        ws = 2 * ws / fs
        N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
        b, a = signal.butter(N, Wn, btype='low', analog=False, output='ba')

        audio = new.T.to("cpu") # torch_lfilter only supports CPU tensor speed up
        a = torch.tensor(a, device="cpu", dtype=torch.float)
        b = torch.tensor(b, device="cpu", dtype=torch.float)
        new_audio = None
        for ppp in range(audio.shape[1]): # torch_lfilter will give weird results for batch samples when using cpu tensor speed up; so we use naive loop here
            new_audio_ = lfilter(b, a, audio[:, ppp:ppp+1]).T
            if new_audio is None:
                new_audio = new_audio_
            else:
                new_audio = torch.cat((new_audio, new_audio_), dim=0)
        new_audio = new_audio.clamp(clip_min, clip_max)
        return new_audio.to(new.device).view(ori_shape)

    def BPF(new, fs=16000, wp=[300, 4000], param=[50, 5000], gpass=3, gstop=40, same_size=True, bits=16):

        assert torch.is_tensor(new) == True
        ori_shape = new.shape
        if len(new.shape) == 1:
            new = new.unsqueeze(0) # (T, ) --> (1, T)
        elif len(new.shape) == 2: # (B, T)
            pass
        elif len(new.shape) == 3:
            new = new.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        if 0.9 * new.max() <= 1 and 0.9 * new.min() >= -1:
            clip_max = 1
            clip_min = -1
            # print(clip_max, clip_min)
        else:
            clip_max = 2 ** (bits - 1) - 1
            clip_min = -2 ** (bits - 1)

        ws = param
        wp = [2 * wp_ / fs for wp_ in wp]
        ws = [2 * ws_ / fs for ws_ in ws]
        N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
        b, a = signal.butter(N, Wn, btype="bandpass", analog=False, output='ba', fs=None)

        audio = new.T.to("cpu")
        a = torch.tensor(a, device="cpu", dtype=torch.float)
        b = torch.tensor(b, device="cpu", dtype=torch.float)

        new_audio = None
        for ppp in range(audio.shape[1]):
            new_audio_ = lfilter(b, a, audio[:, ppp:ppp+1]).T
            if new_audio is None:
                new_audio = new_audio_
            else:
                new_audio = torch.cat((new_audio, new_audio_), dim=0)
        new_audio = new_audio.clamp(clip_min, clip_max)

        return new_audio.to(new.device).view(ori_shape)

    def QT_Non_Diff(audio, param=128, bits=16, same_size=True):

        assert torch.is_tensor(audio) == True
        ori_shape = audio.shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0) # (T, ) --> (1, T)
        elif len(audio.shape) == 2: # (B, T)
            pass
        elif len(audio.shape) == 3:
            audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        max = 2 ** (bits-1) - 1
        min = -1. * 2 ** (bits-1)
        abs_max = abs(min)
        scale = False
        lower = -1
        upper = 1
        # print('QT-1:', audio.max(), audio.min())
        # if audio.min() >= 2 * lower and audio.max() <= 2 * upper: # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
        if 0.9 * audio.max() <= upper and 0.9 * audio.min() >= lower:
            audio = audio * abs_max
            scale = True
        # print('QT-2:', audio.max(), audio.min())
        q = param
        audio_q = torch.round(audio / q) * q # round operation makes it non-differentiable
        # print('QT-3:', audio_q.max(), audio_q.min())

        if scale:
            audio_q.data /= abs_max

        return audio_q.view(ori_shape)

    QT = BPDA(QT_Non_Diff, lambda *args: args[0]) # BPDA wrapper, make it differentiable

    def BDR(audio, param=8, bits=16, same_size=True):
        q = 2 ** (bits - param)
        return QT(audio, param=q, bits=bits, same_size=same_size)

    def AT(audio, param=25, same_size=True):

        assert torch.is_tensor(audio) == True
        ori_shape = audio.shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0) # (T, ) --> (1, T)
        elif len(audio.shape) == 2: # (B, T)
            pass
        elif len(audio.shape) == 3:
            audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        snr = param
        snr = 10 ** (snr / 10)
        batch, N = audio.shape
        power_audio = torch.sum((audio / math.sqrt(N)) ** 2, dim=1, keepdims=True) # (batch, 1)
        power_noise = power_audio / snr # (batch, 1)
        noise = torch.randn((batch, N), device=audio.device) * torch.sqrt(power_noise) # (batch, N)
        noised_audio = audio + noise
        return noised_audio.view(ori_shape)

    def AS(audio, param=3, same_size=True):

        assert torch.is_tensor(audio) == True
        ori_shape = audio.shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0) # (T, ) --> (1, T)
        elif len(audio.shape) == 2: # (B, T)
            pass
        elif len(audio.shape) == 3:
            audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        batch, _ = audio.shape

        kernel_size = param
        assert kernel_size % 2 == 1
        audio = audio.view(batch, 1, -1) # (batch, in_channel:1, max_len)

        ################# Using torch.nn.functional ###################
        kernel_weights = np.ones(kernel_size) / kernel_size
        weight = torch.tensor(kernel_weights, dtype=torch.float, device=audio.device).view(1, 1, -1) # (out_channel:1, in_channel:1, kernel_size)
        output = torch.nn.functional.conv1d(audio, weight, padding=(kernel_size-1)//2) # (batch, 1, max_len)
        ###############################################################

        return output.squeeze(1).view(ori_shape) # (batch, max_len)


    def MS(audio, param=3, same_size=True):
        r"""
        Apply median smoothing to the 1D tensor over the given window.
        """

        assert torch.is_tensor(audio) == True
        ori_shape = audio.shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0) # (T, ) --> (1, T)
        elif len(audio.shape) == 2: # (B, T)
            pass
        elif len(audio.shape) == 3:
            audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        win_length = param
        # Centered windowed
        pad_length = (win_length - 1) // 2

        # "replicate" padding in any dimension
        audio = torch.nn.functional.pad(audio, (pad_length, pad_length), mode="constant", value=0.)

        # indices[..., :pad_length] = torch.cat(pad_length * [indices[..., pad_length].unsqueeze(-1)], dim=-1)
        roll = audio.unfold(-1, win_length, 1)

        values, _ = torch.median(roll, -1)
        return values.view(ori_shape)





    def Speech_Compression_Non_Diff(new, lengths, bits_per_sample,
                                    name, param, fs, same_size,
                                    parallel, n_jobs, start_2, debug):

        def _worker(start_, end):
            st = time.time()
            for i in range(start_, end):
                origin_audio_path = tmp_dir + "/" + str(i) + ".wav"
                audio = np.clip(new[i][:lengths[i]], min, max).astype(np.int16)
                write(origin_audio_path, fs, audio)
                opus_audio_path = "{}/{}.{}".format(tmp_dir, i, name)
                command = "ffmpeg -i {} -ac 1 -ar {} {} {} -c:a {} {}".format(origin_audio_path, fs,
                                param[0], param[1], param[2], opus_audio_path)
                args = shlex.split(command)
                if debug:
                    p = subprocess.Popen(args)
                else:
                    p = subprocess.Popen(args, stderr=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL)
                p.wait()

                pcm_type = "pcm_s16le" if bits_per_sample == 16 else "pcm_s8"
                target_audio_path = tmp_dir + "/" + str(i) + "-target.wav"
                command = "ffmpeg -i {} -ac 1 -ar {} -c:a {} {}".format(opus_audio_path, fs, pcm_type, target_audio_path)
                args = shlex.split(command)
                if debug:
                    p = subprocess.Popen(args)
                else:
                    p = subprocess.Popen(args, stderr=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL)
                p.wait()

                _, coding_audio = read(target_audio_path)
                if coding_audio.size <= lengths[i] or (coding_audio.size > lengths[i] and not same_size):
                    opuseds[i] = list(coding_audio)
                else:
                    start = start_2
                    if start is None:
                        min_dist = np.infty
                        start = 0
                        for start_candidate in range(0, coding_audio.size - audio.size + 1, 1):
                            dist = np.sum(np.abs(audio / abs_max - coding_audio[start_candidate:start_candidate+audio.size] / abs_max))
                            if dist < min_dist:
                                start = start_candidate
                                min_dist = dist
                    opuseds[i] = list(coding_audio[start:start+lengths[i]])
            et = time.time()

        if not bits_per_sample in [16, 8]:
                raise NotImplementedError("Currently We Only Support 16 Bit and 8 Bit Quantized Audio, \
                    You Need to Modify 'pcm_type' for Other Bit Type")

        out_tensor = False
        device = None
        if torch.is_tensor(new):
            device = str(new.device)
            out_tensor = True
            new = new.clone().detach().cpu().numpy()

        ori_shape = new.shape
        if len(new.shape) == 1:
            new = new.reshape((1, new.shape[0])) # (T, ) --> (1, T)
        elif len(new.shape) == 2: # (B, T)
            pass
        elif len(new.shape) == 3:
            new = new.reshape((new.shape[0], new.shape[2])) # (B, 1, T) --> (B, T)
        else:
            raise NotImplementedError('Audio Shape Error')

        bit_rate = param
        n_audios, max_len = new.shape
        ### indicating the real length of each audio in new
        ### this parameter is only valid in speech coding method since other methods not use loop
        lengths = lengths if lengths else n_audios * [max_len]
        max = 2 ** (bits_per_sample-1) - 1
        min = -1. * 2 ** (bits_per_sample-1)
        abs_max = abs(min)
        scale = False
        lower = -1
        upper = 1
        # if -1 <= new.max() <= 1:
        if new.min() >= 2 * lower and new.max() <= 2 * upper: # 2*lower and 2*upper due to floating point issue, e.g., sometimes will have 1.0002
            new = new * abs_max
            scale = True

        high = 100000
        while True:
            random_number = np.random.randint(0, high=high + 1)
            tmp_dir = "{}-Coding-".format(name) + str(random_number)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                break
        opuseds = [0] * n_audios

        if not parallel or (parallel and n_jobs == 1) or n_audios == 1:
            _worker(0, n_audios)

        else:
            n_jobs = n_jobs if n_jobs <= n_audios else n_audios
            n_audios_per_job = n_audios // n_jobs
            process_index = []
            for ii in range(n_jobs):
                process_index.append([ii*n_audios_per_job, (ii+1)*n_audios_per_job])
            if n_jobs * n_audios_per_job != n_audios:
                process_index[-1][-1] = n_audios
            futures = set()
            with ThreadPoolExecutor() as executor:
                for job_id in range(n_jobs):
                    future = executor.submit(_worker, process_index[job_id][0], process_index[job_id][1])
                    futures.add(future)
                for future in as_completed(futures):
                    pass

        shutil.rmtree(tmp_dir)

        ##change to solve AMR problem
        #opuseds = np.array([(x+[0]*(max_len-len(x)))[:max_len] for x in opuseds])
        opuseds = np.array([(x if isinstance(x, list) else [x]) + [0] * (max_len - len(x if isinstance(x, list) else [x])) for x in opuseds])

        opuseds = opuseds.reshape(ori_shape)
        if out_tensor:
            opuseds = torch.tensor(opuseds, dtype=torch.float, device=device)
        if scale:
            opuseds.data /= abs_max
        return opuseds

    global speech_compression
    speech_compression = BPDA(Speech_Compression_Non_Diff, lambda *args: args[0])

    def OPUS(new, lengths=None, bits_per_sample=16, param=16000, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):


        return speech_compression(new, lengths, bits_per_sample,
                'opus', ['-b:a', param, 'libopus'],
                fs, same_size,
                parallel, n_jobs, 69, debug)


    def SPEEX(new, lengths=None, bits_per_sample=16, param=43200, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

        return speech_compression(new, lengths, bits_per_sample,
                'spx', ['-b:a', param, 'libspeex'],
                fs, same_size,
                parallel, n_jobs, None, debug)


    def AMR(new, lengths=None, bits_per_sample=16, param=6600, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

        if fs == 16000:
            legal_bit_rate = [6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850]
        elif fs == 8000:
            legal_bit_rate = [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]
        else:
            raise NotImplementedError("AMR Compression only support sampling rate 16000 and 8000")
        if not int(param) in legal_bit_rate:
            raise NotImplementedError("%f Not Allowed When fs=%d" % (param, fs))

        return speech_compression(new, lengths, bits_per_sample,
                'amr', ['-b:a', param, "libvo_amrwbenc" if fs == 16000 else "libopencore_amrnb"],
                fs, same_size,
                parallel, n_jobs, None, debug)


    def AAC_V(new, lengths=None, bits_per_sample=16, param=5, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

        return speech_compression(new, lengths, bits_per_sample,
                'aac', ['-vbr', param, 'libfdk_aac'],
                fs, same_size,
                parallel, n_jobs, 2048, debug)


    def AAC_C(new, lengths=None, bits_per_sample=16, param=20000, fs=16000, same_size=True, parallel=True, n_jobs=10, debug=False):

        return speech_compression(new, lengths, bits_per_sample,
                'aac', ['-b:a', param, 'libfdk_aac'],
                fs, same_size,
                parallel, n_jobs, 2048, debug)


    def MP3_V(new, lengths=None, param=9, fs=16000, bits_per_sample=16, same_size=True, parallel=True, n_jobs=10, debug=False):

        return speech_compression(new, lengths, bits_per_sample,
                'mp3', ['-q:a', param, 'mp3'],
                fs, same_size,
                parallel, n_jobs, 0, debug)


    def MP3_C(new, lengths=None, param=16000, fs=16000, bits_per_sample=16, same_size=True, parallel=True, n_jobs=10, debug=False):

        return speech_compression(new, lengths, bits_per_sample,
                'mp3', ['-b:a', param, 'mp3'],
                fs, same_size,
                parallel, n_jobs, 0, debug)


class DefenseEvaluator:

    def __init__(self, model, defenses, clean_samples, clean_labels, adv_samples, adv_labels, device='cpu'):
        self.model = model.to(device)
        self.defenses = defenses  # dict {'type_of_defense':[defense1, defense2..]} #Preprocessing #Detector #AdversarialTraining
        self.device = device
        self.clean_samples = clean_samples
        self.clean_labels = clean_labels
        self.clean_samples_tensor = self.get_tensor(self.clean_samples)
        self.clean_labels_tensor = self.get_tensor(self.clean_labels)
        self.adv_samples = adv_samples
        self.adv_labels = adv_labels
        self.adv_samples_tensor = self.get_tensor(self.adv_samples)
        self.adv_labels_tensor = self.get_tensor(self.adv_labels)
        self.mixed_samples = self.clean_samples + self.adv_samples
        self.mixed_samples_tensor = self.get_tensor(self.mixed_samples)
        self.mixed_labels = self.clean_labels + self.adv_labels
        self.mixed_labels_tensor = self.get_tensor(self.mixed_labels)

    def generate_parameters_dicts(self,params_dict):
        if not params_dict:  # If the params_dict is empty, yield an empty dict
            yield {}
            return

        keys, values = zip(*params_dict.items())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def apply_defenses(self):
        results = []
        #print("cs",self.clean_samples,"\n cl",self.clean_labels)
        base_clean_accuracy = self.evaluate(self.clean_samples,self.clean_labels)
        base_adv_accuracy = self.evaluate(self.adv_samples,self.adv_labels)
        base_mixed_accuracy = self.evaluate(self.mixed_samples,self.mixed_labels)
        base_eer_clean,_ = compute_eer_samples(self.model,self.clean_samples_tensor,self.clean_labels_tensor)
        base_eer_mixed,_ = compute_eer_samples(self.model,self.mixed_samples_tensor,self.mixed_labels_tensor)

        print(f"Model base accuracy on clean samples: {base_clean_accuracy}")
        print(f"Model base accuracy on mixed samples: {base_mixed_accuracy}")
        print(f"Model base accuracy on adversarial samples: {base_adv_accuracy}")
        print(f"EER on clean: {base_eer_clean}")
        print(f"EER on mixed: {base_eer_mixed}")
        results.append({
        "Defense":str(self.model),
        "Type": "None",
        'Accuracy on Clean Samples': base_clean_accuracy,
        'Accuracy on Adversarial Samples': base_adv_accuracy,
        'Accuracy on Mixed Samples': base_mixed_accuracy,
        'EER clean samples': base_eer_clean,
        'EER mixed samples':base_eer_mixed,
        })

        for defense_type, defenses in self.defenses.items():
            if defense_type == "preprocessing":
                for defense_dict in defenses:
                    defense = defense_dict['defense']
                    params_list = defense_dict.get('params', {})  # Use an empty dict if 'params' is missing
                    best_param_value = float('inf')  # Initialize with a high value
                    best_combination = None
                    best_processed_metrics = {}

                    # Check if there are parameters to generate combinations
                    if params_list:
                        param_combinations = self.generate_parameters_dicts(params_list)
                    else:
                        param_combinations = [{}]  # Use a single empty dict to indicate no parameters

                    print(param_combinations)
                    # Generate all combinations of parameters, or a single iteration for no parameters
                    for param_combination in param_combinations:
                        print(f"Applying defense {defense} with params {param_combination}")


                        # Apply defense with current parameter combination
                        if param_combination:  # If there are parameters, unpack them
                            processed_clean_samples = defense(self.clean_samples_tensor, **param_combination)
                            processed_adv_samples = defense(self.adv_samples_tensor, **param_combination)
                        else:  # If there are no parameters, call the defense without them
                            processed_clean_samples = defense(self.clean_samples_tensor)
                            processed_adv_samples = defense(self.adv_samples_tensor)

                        ##
                        '''
                        processed_clean_samples = []
                        processed_adv_samples = []

                        # Iterate over clean samples
                        for i in range(len(self.clean_samples_tensor)):
                            # Extract the current clean sample
                            current_clean_sample = self.clean_samples_tensor[i:i+1]  # Keep the sample in a batch-like format for compatibility

                            # Apply defense with current parameter combination to the clean sample
                            if param_combination:  # If there are parameters, unpack them
                                processed_sample = defense(current_clean_sample, **param_combination)
                            else:  # If there are no parameters, call the defense without them
                                processed_sample = defense(current_clean_sample)

                            # Append the processed clean sample to the accumulator
                            processed_clean_samples.append(processed_sample)

                        # Similarly, iterate over adversarial samples
                        for j in range(len(self.adv_samples_tensor)):
                            # Extract the current adversarial sample
                            current_adv_sample = self.adv_samples_tensor[j:j+1]  # Keep the sample in a batch-like format for compatibility

                            # Apply defense with current parameter combination to the adversarial sample
                            if param_combination:  # If there are parameters, unpack them
                                processed_sample = defense(current_adv_sample, **param_combination)
                            else:  # If there are no parameters, call the defense without them
                                processed_sample = defense(current_adv_sample)

                            # Append the processed adversarial sample to the accumulator
                            processed_adv_samples.append(processed_sample)

                        # Convert lists to tensors if necessary
                        processed_clean_samples = torch.cat(processed_clean_samples, dim=0)
                        processed_adv_samples = torch.cat(processed_adv_samples, dim=0)
                        # Adjust dimensions of processed samples to [513, 1, 32000]
                        if processed_clean_samples.dim() == 2:
                            processed_clean_samples = processed_clean_samples.unsqueeze(1)  # Add a dimension at position 1
                            processed_adv_samples = processed_adv_samples.unsqueeze(1)  # Add a dimension at position 1

                        '''#
                        '''
                        num_samples_clean = self.clean_samples_tensor.size(0)  # Assuming samples are the first dimension
                        indices_clean = random.sample(range(num_samples_clean), min(1, num_samples_clean))  # Select 10 random indices or less if fewer samples

                        index = min(1,len(str(defense).split())-1)
                        defense_name = str(defense).split()[index]
                        for i in indices_clean:
                            original_sample = self.clean_samples_tensor[i].squeeze().cpu().numpy()
                            processed_sample = processed_clean_samples[i].squeeze().cpu().numpy()
                            #print(processed_sample)
                            if isinstance(processed_sample, torch.Tensor):
                                processed_sample = processed_sample.squeeze().cpu().numpy()  # Squeeze to remove any extra dimensions

                            plt.figure(figsize=(12, 6))
                            plt.subplot(1, 2, 1)
                            plt.plot(np.linspace(0, 1, len(original_sample)), original_sample, label=f'Original Clean {i}')
                            plt.title(f"Original Clean Sample {i} Waveform")
                            plt.xlabel('Time')
                            plt.ylabel('Amplitude')
                            plt.legend()

                            plt.subplot(1, 2, 2)
                            plt.plot(np.linspace(0, 1, len(processed_sample)), processed_sample, label=f'Processed Clean {i}')
                            plt.title(f"Processed with {defense_name} Clean Sample {i} Waveform")
                            plt.xlabel('Time')
                            plt.ylabel('Amplitude')
                            plt.legend()

                            plt.show()

                        # Plot for adversarial samples
                        num_samples_adv = self.adv_samples_tensor.size(0)  # Assuming samples are the first dimension
                        indices_adv = random.sample(range(num_samples_adv), min(10, num_samples_adv))  # Select 10 random indices or less if fewer samples

                        for i in indices_adv:
                            original_adv_sample = self.adv_samples_tensor[i].squeeze().cpu().numpy()
                            processed_adv_sample = processed_adv_samples[i].squeeze().cpu().numpy()

                            plt.figure(figsize=(12, 6))
                            plt.subplot(1, 2, 1)
                            plt.plot(np.linspace(0, 1, len(original_adv_sample)), original_adv_sample, label=f'Original Adv {i}')
                            plt.title(f"Original Adv Sample {i} Waveform")
                            plt.xlabel('Time')
                            plt.ylabel('Amplitude')
                            plt.legend()

                            plt.subplot(1, 2, 2)
                            plt.plot(np.linspace(0, 1, len(processed_adv_sample)), processed_adv_sample, label=f'Processed Adv {i}')
                            plt.title(f"Processed with {defense_name} Adv Sample {i} Waveform")
                            plt.xlabel('Time')
                            plt.ylabel('Amplitude')
                            plt.legend()

                            plt.show()
                        ##
                        '''
                        processed_clean_accuracy = self.evaluate(processed_clean_samples, self.clean_labels)
                        processed_adv_accuracy = self.evaluate(processed_adv_samples, self.adv_labels)
                        processed_mixed_samples = torch.cat((processed_clean_samples, processed_adv_samples), dim=0)
                        processed_mixed_accuracy = processed_clean_accuracy * (len(processed_clean_samples)/(len(processed_clean_samples)+len(processed_adv_samples))) + processed_adv_accuracy * (len(processed_adv_samples)/(len(processed_clean_samples)+len(processed_adv_samples)))
                        #processed_mixed_accuracy =  self.evaluate(processed_mixed_samples, self.mixed_labels)

                        eer_clean, _ = compute_eer_samples(self.model, processed_clean_samples, self.clean_labels_tensor)
                        eer_mixed, _ = compute_eer_samples(self.model, processed_mixed_samples, self.mixed_labels_tensor)

                        param_value = eer_clean * 0.75+ eer_mixed * 0.25

                        print(f"Defense {defense} with params {param_combination} has EER mixed: {eer_mixed}")
                        # Check if this combination has the best EER on mixed samples
                        if param_value < best_param_value:
                            best_eer_mixed = eer_mixed
                            best_combination = param_combination
                            best_processed_metrics = {
                                'Defense': str(defense),
                                'Type': defense_type,
                                'Params': best_combination,
                                'Accuracy on Clean Samples': processed_clean_accuracy,
                                'Accuracy on Adversarial Samples': processed_adv_accuracy,
                                'Accuracy on Mixed Samples': processed_mixed_accuracy,
                                'EER clean samples': eer_clean,
                                'EER mixed samples': eer_mixed,
                            }

                    # Add the best combination and its metrics to the results
                    #if best_combination:
                    print(f"Best combination for {defense} is {best_combination} with EER on mixed samples: {best_eer_mixed}")
                    results.append(best_processed_metrics)

            else:# defense_type in ['adversarial_training', 'detector','detector+adversarial_training','transferabilty']:
                for defense_dict in defenses:

                    defense = defense_dict['defense']
                    defense = defense.to(device)
                    print(f"Applying {defense_type} {defense}")
                    clean_accuracy = self.evaluate(self.clean_samples, self.clean_labels, model=defense)
                    adv_accuracy = self.evaluate(self.adv_samples, self.adv_labels, model=defense)
                    mixed_accuracy = self.evaluate(self.mixed_samples_tensor,self.mixed_labels,model = defense)
                    print(f"Accuracy {defense_type} on clean samples {clean_accuracy}")
                    print(f"Accuracy {defense_type} on adv samples {adv_accuracy}")
                    print(f"Accuracy {defense_type} on mixed samples {mixed_accuracy}")

                    eer_clean, _ = compute_eer_samples(defense, self.clean_samples_tensor, self.clean_labels_tensor)
                    #processed_clean_samples = defense(self.clean_samples_tensor) if defense_type == 'detector' else self.clean_samples_tensor
                    #processed_adv_samples = defense(self.adv_samples_tensor) if defense_type == 'detector' else self.adv_samples_tensor
                    eer_mixed, _ = compute_eer_samples(defense, self.mixed_samples_tensor, self.mixed_labels_tensor)
                    print(f"EER on clean: {eer_clean}")
                    print(f"EER on mixed: {eer_mixed}")
                    defense.to('cpu')
                    results.append({
                        'Defense': str(defense),
                        'Type': defense_type,
                        'Accuracy on Clean Samples': clean_accuracy,
                        'Accuracy on Adversarial Samples': adv_accuracy,
                        'Accuracy on Mixed Samples': mixed_accuracy,
                        'EER clean samples': eer_clean,
                        'EER mixed samples': eer_mixed,
                    })


        return results

    def predict(self, audio, model = None):
        # Ensure audio is in the correct format (list or array-like of numbers)
        #print(type(audio),type(audio[0]))
        if model is None:
            model = self.model

        audio = self.get_tensor(audio)
        audio = audio.to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = model(audio)
            _, predicted = torch.max(outputs.data, 1)
        return predicted



    def evaluate(self, audio, labels, model=None):
        #print("cs",self.clean_samples,"\n cl",self.clean_labels)
        predicted = self.predict(audio, model)
        #print(type(audio), type(audio[0]), type(labels), type(labels[0]),print(labels),print(labels[0]))

        # Ensure labels is a flat list of integers
        labels = [int(label) for label in labels]

        # Convert labels to a NumPy array and then to a tensor
        labels_np = np.array(labels, dtype=np.int64)
        labels_tensor = torch.tensor(labels_np, dtype=torch.long, device=self.device)

        accuracy = accuracy_score(labels_tensor.cpu(), predicted.cpu())
        return accuracy


    def get_tensor(self, samples):
        if not isinstance(samples, torch.Tensor):
            if isinstance(samples, list):
                if all(isinstance(sample, torch.Tensor) for sample in samples):
                    # Stack tensors if samples is a list of tensors
                    samples = torch.stack(samples, dim=0).clone().detach()
                elif all(isinstance(sample, np.ndarray) for sample in samples):
                    # Convert list of numpy arrays to tensors and stack
                    samples = torch.stack([torch.tensor(sample, dtype=torch.float) for sample in samples], dim=0)
                elif all(isinstance(sample, list) for sample in samples):
                    # Convert list of lists to tensors and stack
                    samples = torch.stack([torch.tensor(sample, dtype=torch.float) for sample in samples], dim=0)
                else:
                    # Handle mixed list of numpy arrays and tensors
                    samples = torch.stack([sample if isinstance(sample, torch.Tensor) else torch.tensor(sample, dtype=torch.float) for sample in samples], dim=0)
            else:
                # Directly convert to tensor
                samples = torch.tensor(samples, dtype=torch.float)

        # Move tensor to the same device as the model
        samples = samples.to(self.device)

        return samples
