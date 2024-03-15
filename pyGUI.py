import PySimpleGUI as sg
import sounddevice as sd
import tempfile
import numpy as np
import os
import torch
from torch import nn,optim
import torchaudio
import soundfile as sf
import threading
from Models import DeepSpeech,CombinedDetectorModel
import queue  # Make sure to import queue at the top of your script
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from Attacks import *
import art
from art.estimators.classification import PyTorchClassifier
from Dataset import *
import pygame 
from Evaluation import *
import glob

pygame.mixer.init()



def play_audio(audio_file):
    if os.path.exists(audio_file):
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
    else:
        print(f"Audio file not found: {audio_file}")


def combine_audio_chunks(chunks, original_length=None):

    # Ensure chunks are all PyTorch tensors
    #print(chunks[0])
    print([torch.tensor(chunk).size() for chunk in chunks])
    tensor_chunks = [torch.tensor(chunk) if not isinstance(chunk, torch.Tensor) else chunk for chunk in chunks]

    combined_signal = torch.cat(tensor_chunks, dim=-1)

    print("combined signal",combined_signal.size())

    if combined_signal.dim()>2:
        combined_signal = combined_signal.squeeze(0)

    print("combined signal",combined_signal.size())

    if original_length is not None and combined_signal.size(1) > original_length:
        print("ol",original_length)
        combined_signal = combined_signal[:, :original_length]
    print("combined signal",combined_signal.size())
    
    return combined_signal

def divide_audio_into_chunks(audio, chunk_size_seconds=2, sr=16000,pad = True):

    if audio.size(0) > 1:  # Check if stereo and convert to mono if necessary
        audio = torch.mean(audio, dim=0).unsqueeze(0)

    samples_per_chunk = sr * chunk_size_seconds
    n_chunks = (audio.size(1) // samples_per_chunk) + (audio.size(1) % samples_per_chunk > 0)
    chunks = []

    for i in range(n_chunks):
        start_sample = i * samples_per_chunk
        end_sample = start_sample + samples_per_chunk
        chunk = audio[:, start_sample:end_sample]

        if chunk.shape[1] < samples_per_chunk and pad:
            pad_size = samples_per_chunk - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))

        chunks.append(chunk)
    
    return chunks



def predict_audio(audio, model, chunk_size_seconds=2, sr=16000,pad = True):
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio).float()
    device = next(model.parameters()).device
    signal = audio.to(device)

    #print("audio",signal.size())
          
    if signal.size(0) > 1:  # Check if stereo and convert to mono if necessary
        signal = torch.mean(signal, dim=0).unsqueeze(0)
    #if signal.dim()>2:
    #    signal = signal.view(signal.size(0),signal.size(1))
    
    chunks = divide_audio_into_chunks(signal, chunk_size_seconds, sr,pad)
    #print("audio",signal.size(),"palen",len(chunks))

    preds = []
    weights = []

    for chunk in chunks:
        chunk = chunk.to(device)
        with torch.no_grad():  # Disable gradients for inference
            model_preds = model(chunk)

        preds.append(model_preds.detach())
        weight = chunk.shape[1] / (sr * chunk_size_seconds)
        weights.append(weight)

    preds_tensor = torch.cat(preds, dim=0)
    weights_tensor = torch.tensor(weights).unsqueeze(1).to(preds_tensor.device)
    weighted_mean_preds = torch.sum(preds_tensor * weights_tensor, dim=0) / torch.sum(weights_tensor)

    final_label = torch.argmax(weighted_mean_preds).item()
    return final_label
    


def predict_audio_path(audio_path, model, chunk_size_seconds=2):
    print(audio_path)
    signal, sr = torchaudio.load(audio_path)
    return predict_audio(signal,model,chunk_size_seconds,sr)

    
# Placeholder predict function
def predict(model, filepath):
    final_label = predict_audio_path(filepath, model,chunk_size_seconds=2)
    return str(final_label)
    
    
# Load your model
def load_base_model(path,n_feature,num_classes,device):
    # Initialize and load your model here
    model = DeepSpeech(n_feature=n_feature, n_hidden=512, n_class=num_classes, dropout=0.35)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set the model to evaluation mode
    return model


def load_attacks(pkl_names, directory='./Models_saved'):
    attacks = {}
    for pkl_name in pkl_names:
        full_path = os.path.join(directory, pkl_name)
        #try:
        #    with open(full_path, 'rb') as file:
        #        attack_name = pkl_name.replace('_attack.pkl', '')
        #        attacks[attack_name] = pickle.load(file)
        try:
            # Specify map_location to 'cpu' if CUDA is not available
            attack_obj = torch.load(full_path, map_location=lambda storage, loc: storage)
            
            attack_name = pkl_name.replace('_attack.pkl', '')
            attacks[attack_name] = attack_obj

        except FileNotFoundError:
            print(f"File {full_path} not found. Skipping...")
        except Exception as e:
            print(f"Failed to load {full_path}: {e}")
    return attacks
    
def save_synthetic_samples(synthetic_samples, audio_samples_path):
    if not os.path.exists(audio_samples_path):
        os.makedirs(audio_samples_path)
    for attack_name, samples in synthetic_samples.items():
        for i, sample in enumerate(samples):
            if i == 0:
                filename = os.path.join(audio_samples_path, f"{attack_name}.wav")
            else:
                filename = os.path.join(audio_samples_path, f"{attack_name}_{i}.wav")
            torchaudio.save(filename, sample, 16000)  # Assuming 16000Hz as sample rate

def load_synthetic_samples(audio_samples_path):
    loaded_samples = {}
    for filepath in glob.glob(os.path.join(audio_samples_path, '*.wav')):
        attack_name = os.path.basename(filepath).split('_')[0]
        if attack_name not in loaded_samples:
            loaded_samples[attack_name] = []
        sample, sr = torchaudio.load(filepath)
        loaded_samples[attack_name].append(sample)
    return loaded_samples

    
def generate_synthetic_samples(audio_path,audio_samples_path,model_to_attack,pytorch_classifier):
    
    signal, sr = torchaudio.load(audio_path)
    signal_length = max(signal.shape)
    art_attacks_names = ['FGSM','BIM','PGD','DF','CW']
    art_pkl_names = [name+'_attack.pkl' for name in art_attacks_names]

    fgsm_attack = ART_FGSM(pytorch_classifier)
    bim_attack = ART_BIM(pytorch_classifier)
    pgd_attack = ART_PGD(pytorch_classifier)
    df_attack = ART_DF(pytorch_classifier)
    cw_attack = ART_CW(pytorch_classifier)

    art_attacks = {
        'FGSM': fgsm_attack,
        'BIM': bim_attack,
        'PGD': pgd_attack,
        'DF': df_attack,
        'CW': cw_attack,
    }

    #art_attacks = load_attacks(art_pkl_names)

    max_value = 1
    min_value = 0#-1
    range_values = round(max_value - min_value,3)
    steps = 3#5
    num_steps = steps + 1
    e_step = round(range_values/steps,3)
    max_v = range_values
    min_v = round(max_value/steps,3)

    fgsm_params = {
        'eps': np.round(np.linspace(min_v, max_v, num_steps), 3),
        'targeted': [True]
    }
    bim_params = {
        'eps': np.round(np.linspace(min_v, max_v, num_steps), 3),
        'eps_step': np.round(np.linspace(min_v / steps, min_v, num_steps), 3),
        'max_iter': [5],
        'targeted': [True],
        'verbose':[False],
    }
    pgd_params = bim_params
    df_params = {
        'max_iter': [5,10,25],
        'epsilon': [1e-3,1e-2,1],
        'nb_grads': [2],
        'batch_size': [32]
    }
    cw_params = {
        'confidence': [0.25],
        'targeted': [True],
        'learning_rate': [0.25],
        'binary_search_steps': [5],
        'max_iter': [10],
        'initial_const': [0.01],
        'max_halving': [4],
        'max_doubling': [4],
        'batch_size': [32],
        'verbose':[True],
    }
    # Generate parameter combinations
    fgsm_combinations = ART_Attacks.generate_parameters_dicts(fgsm_params)
    bim_combinations = ART_Attacks.generate_parameters_dicts(bim_params)
    pgd_combinations = ART_Attacks.generate_parameters_dicts(pgd_params)
    df_combinations = ART_Attacks.generate_parameters_dicts(df_params)
    cw_combinations = ART_Attacks.generate_parameters_dicts(cw_params)

    art_combinations = {
    'FGSM': fgsm_combinations,
    'BIM': bim_combinations,
    'PGD': pgd_combinations,
    'DF': df_combinations,
    'CW': cw_combinations
    }
    attacked_model = model_to_attack

    chunks = divide_audio_into_chunks(signal, chunk_size_seconds = 2, sr = 16000)
    
    # First, ensure all chunks are tensors; this should already be the case
    chunks = [torch.tensor(chunk, dtype=torch.float32) for chunk in chunks]
    

    # Create a tensor of zeros with length equal to the number of chunks
    # Adjust dtype according to what `MultipleARTAttacks` expects for this parameter
    labels = torch.zeros(len(chunks), dtype=torch.long)

    multiple_art_attacks = MultipleARTAttacks(attacked_model, art_attacks, art_combinations, torch.stack(chunks), labels)
    multiple_art_attacks.run_attacks(N_top = 1)
    multiple_art_attacks.analyze_results()
    results = multiple_art_attacks.get_results()

    synthetic_samples = {}
    

    
    for attack_name,result in results[attacked_model].items():
        top_params = result['params']
        top_samples = result['samples']
        
        synthetic_samples[attack_name] = []
        for param,samples in zip(top_params,top_samples):
            #print("len samples",len(samples))
            synthetic_sample = combine_audio_chunks(samples, original_length=signal_length)
            #print("len syn samp",len(synthetic_sample),synthetic_sample.size())
            synthetic_samples[attack_name].append(synthetic_sample)

    fakebob_params = {'model': [attacked_model], 'confidence':[0.5],'threshold': [None],'task':['CSI'],'targeted': [True], 'epsilon': [0.00001], 'max_iter':[50], 'max_lr': [0.5], 'verbose': [0],'batch_size':[1] }
    sirenattack_params = {'model':[attacked_model], 'confidence':[0.5],'verbose':[0],'epsilon': [0.00001],'max_iter':[50],'batch_size':[32]}
    kenan_params = {'model': [attacked_model], 'atk_name':['fft'],'max_iter' : [50],'verbose': [0],'early_stop': [True],'targeted':[True],'batch_size':[1]}

    sg_combinations = {
                    "SIRENATTACK":sirenattack_params,
                    "FAKEBOB":fakebob_params,
                    "KENAN":kenan_params,
                    }

    fakebob_attack = SG_FAKEBOB()
    sirenattack_attack = SG_SIRENATTACK()
    kenan_attack = SG_KENAN()

    sg_attacks = {

            "SIRENATTACK":sirenattack_attack,
            "FAKEBOB":fakebob_attack,
            "KENAN":kenan_attack,
                }
    
    multipleSGattacks = MultipleSGAttacks(model=model_to_attack, attacks=sg_attacks, combinations=sg_combinations, samples=torch.stack(chunks), labels=labels)
    #print([max(chunk) for chunk in chunks])
    multipleSGattacks.run_attacks(N_top = 1)
    multipleSGattacks.analyze_results()
    results = multipleSGattacks.get_results()

    for attack_name,result in results[attacked_model].items():
        top_params = result['params']
        top_samples = result['samples']
        
        synthetic_samples[attack_name] = []
        for param,samples in zip(top_params,top_samples):
            #print("len samples",len(samples))
            #print(samples[0].size())
            samples = [sample.tolist() for sample in samples]
            synthetic_sample = combine_audio_chunks(samples, original_length=signal_length)
            #print("len syn samp",len(synthetic_sample),synthetic_sample.size())

            synthetic_samples[attack_name].append(synthetic_sample)
        
    return synthetic_samples







class AudioRecorder:
    def __init__(self, fs=16000):
        self.fs = fs
        #self.prevq = None
        self.q = queue.Queue()
        self.recording = False
        self.filepath = None

    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def start(self):
        if not self.recording:
            self.recording = True
            self.stream = sd.InputStream(samplerate=self.fs, channels=2, callback=self.callback)
            self.stream.start()

    def stop(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self._save()

    def _save(self):
        self.filepath = tempfile.mktemp(prefix='recorded_', suffix='.wav', dir='.')
        #self.prevq = self.q.copy()
        with sf.SoundFile(self.filepath, mode='x', samplerate=self.fs, channels=2) as file:
            while not self.q.empty():
                file.write(self.q.get())


def runner(cmd):
     os.system(cmd)




def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    chunk_size = 2
    n_feature = sr*chunk_size
    num_classes = 2
    deepspeech_model = DeepSpeech(n_feature = n_feature, n_hidden = 512, n_class = num_classes, dropout= 0.35)
    #base_path = "/home/domenico/Desktop/MAGISTRALE/Tesi/Git_20_2"
    base_path = os.getcwd()
    dataset_path = os.path.join(base_path,"SpReW")

    task = "SV"
    sprew_path = os.path.join(base_path,"sprew.pkl")
    if not os.path.exists(sprew_path):
        dataset = ChunkedSpReWDataset(base_path = dataset_path,task = task,chunk_size = chunk_size,generate_signals=True)
        with open(sprew_path, 'wb') as file:
            pickle.dump(dataset,file)
    else:
        with open(sprew_path, 'rb') as file:
            dataset = pickle.load(file)
    #training
    tr_paths = dataset.get_splitted_paths()['training']
    val_paths = dataset.get_splitted_paths()['validation']
    test_paths = dataset.get_splitted_paths()['test']

    splitted_signals = dataset.get_splitted_signals()

    signals_tr = splitted_signals['training']
    signals_val = splitted_signals['validation']
    signals_test = splitted_signals['test']

    labels_tr = [dataset.get_label_from_path(path) for path in tr_paths]
    labels_val = [dataset.get_label_from_path(path) for path in val_paths]
    labels_test = [dataset.get_label_from_path(path) for path in test_paths]

    test_loader = to_loader(signals_test, labels_test)


    models_path = os.path.join(base_path,"Models_saved","2024_3_12")
    model_save_path_deepspeech = os.path.join(models_path,'model_DeepSpeech.pt')

    #loading defended model
    detector = DeepSpeech(n_feature = n_feature, n_hidden = 512, n_class = num_classes, dropout= 0.35)
    model_save_path_detector = os.path.join(models_path,'model_Detector_DeepSpeech.pt')
    state_dict = torch.load(model_save_path_detector,map_location=device)
    missing_keys, unexpected_keys = detector.load_state_dict(state_dict, strict=False)
    print("Loaded pre-trained model: ", model_save_path_detector)

    #loading defended model
    adversarial_deepspeech = DeepSpeech(n_feature = n_feature, n_hidden = 512, n_class = num_classes, dropout= 0.35)
    model_save_path_adversarial = os.path.join(models_path,'model_Adversarial_DeepSpeech.pt')
    state_dict = torch.load(model_save_path_adversarial ,map_location=device)
    missing_keys, unexpected_keys = adversarial_deepspeech.load_state_dict(state_dict, strict=False)
    print("Loaded pre-trained model: ", model_save_path_adversarial)


    pytorch_classifier = PyTorchClassifier(
                                        model=deepspeech_model,
                                        clip_values=(-1,1),
                                        loss= nn.CrossEntropyLoss(),
                                        optimizer=optim.Adam(deepspeech_model.parameters(), lr=1e-5),
                                        input_shape=32000,
                                        nb_classes=num_classes

                                    )

    x_tr_tensor = torch.stack(signals_tr+signals_val)
    labels_tr_tensor = torch.Tensor(labels_tr+labels_val)
    x_tr_cpu = x_tr_tensor.cpu()
    labels_tr_cpu = labels_tr_tensor.cpu()

    #Trained on training and val samples
    nb_epochs = 5
    pytorch_classifier.fit(x_tr_tensor, labels_tr_tensor, batch_size=32, nb_epochs=nb_epochs, verbose=True)


    # Initialize your model
    deepspeech_model = load_base_model(model_save_path_deepspeech,n_feature=n_feature,num_classes = num_classes,device = device)
    detector_deepspeech = CombinedDetectorModel(detector,deepspeech_model)

    print("test deep: ",torch_test_phase(deepspeech_model,test_loader= test_loader,criterion= get_criterion(num_classes=num_classes)))
    print("test det+deep: ",torch_test_phase(detector_deepspeech,test_loader= test_loader,criterion= get_criterion(num_classes=num_classes)))
    print("test adv: ",torch_test_phase(adversarial_deepspeech,test_loader= test_loader,criterion= get_criterion(num_classes=num_classes)))

    models = {deepspeech_model:'deepspeech',detector_deepspeech:'detector',adversarial_deepspeech:'adversarial'}
    

    audio_recorder = AudioRecorder()

    # pysimpleGUI INIT:
    AppFont = 'Any 16'
    sg.theme('DarkTeal2')

    buttons = [
        [   
            sg.FileBrowse("Browse Audio File", file_types=(("Audio Files", "*.wav"),), key='-BROWSE-', target='-AUDIO_PATH-'),
            sg.Button("Record", key='-RECORD-', font=AppFont), 
            sg.Button("Predict", font=AppFont),
            sg.Text("Select desired Audio Model:"),
            #sg.Listbox(list(models.values()), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, key='-MODEL-', size=(20, 6), enable_events=True),
            sg.Listbox(list(models.values()), select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, key='-MODEL-', size=(18, 5), enable_events=True, default_values=list(models.values())),
            sg.Button('ShowMic', font=AppFont), 
            sg.Button('Exit', font=AppFont)
        ],
        [
            #sg.FileBrowse("Browse Audio Files", file_types=(("Audio Files", "*.wav"),), key='-BROWSE-', target='-AUDIO_PATH-'),
            sg.In(visible=False, enable_events=True, key='-AUDIO_PATH-'),
            sg.Button("Play Audio", key='-PLAY_AUDIO-', font=AppFont),
            sg.Table(values=[], headings=['Model', 'Prediction'], key='-TABLE-', display_row_numbers=False, auto_size_columns=False, num_rows=10, def_col_width=17, justification='center'),
            sg.Table(values=[], headings=['Model', 'Attack', 'Prediction','Perturbation','SNR'], key='-TABLE_ATTACKS-', display_row_numbers=False, auto_size_columns=False, num_rows=10, def_col_width=15, justification='center'),
            sg.Listbox(values=[], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='-GENERATED_AUDIOS-', size=(18, 5), enable_events=True),
            sg.Button("Play Selected Attack Audio", key='-PLAY_SELECTED_AUDIO-', font=AppFont),
        ],
        [sg.Button("Attack", key='-ATTACK-', font=AppFont)]

    ]


    # VARS CONSTS:
    _VARS = {'window': sg.Window('Audio Recorder and Predictor',layout = [buttons], finalize=True,location=(400, 100), element_justification='c'),
            'stream': False,
            'audioPath': None,
            'syntheticSamplesPath': os.path.join(os.getcwd(),'synthetic_samples')
            }
    
    if not os.path.exists(_VARS['syntheticSamplesPath']):
        os.makedirs(_VARS['syntheticSamplesPath'])


    temp_filename=None

    while True:
        event, values = _VARS['window'].read(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == '-RECORD-':
            if not audio_recorder.recording:
                _VARS['window']['-RECORD-'].update('Stop')
                audio_recorder.start()
            else:
                _VARS['window']['-RECORD-'].update('Record')
                audio_recorder.stop()
                # No need to join threads since recording is handled internally
                temp_filename = audio_recorder.filepath  # Use the filepath from AudioRecorder
                audio_file_name = temp_filename
                _VARS['window']['-TABLE-'].update(values=[["Audio",audio_file_name]])

        elif event == 'Predict':

            audio_path = values['-AUDIO_PATH-'] if values['-AUDIO_PATH-'] else audio_recorder.filepath
            

            if audio_path and os.path.exists(audio_path):
                selected_model_names = values['-MODEL-']  # This will now be a list
                predictions = []
                audio_file_name = os.path.basename(audio_path)
                predictions.append(["Audio",audio_file_name])
                row_colors = []

                for i,model_name in enumerate(selected_model_names,start = 1):
                    name_to_model = {v: k for k, v in models.items()}
                    selected_model = name_to_model.get(model_name)
                    if selected_model:
                        label = predict_audio_path(audio_path, selected_model)
                        #print(f"model {selected_model} predict {label}")
                        prediction = 'Authorized' if label == 1 else 'Not Authorized'
                        predictions.append([model_name, prediction])
                        # Decide the row color based on the prediction
                        color = 'green' if prediction == 'Authorized' else 'red'
                        row_colors.append((i, color)) 
                    else:
                        sg.popup(f"Model {model_name} not found.")
                _VARS['window']['-TABLE-'].update(values=predictions, row_colors=row_colors)
            else:
                sg.popup("Please record audio or select a file first.")
            
        elif event == '-BROWSE-':
            # Update the audio path when a file is selected
            _VARS['audioPath'] = values['-AUDIO_PATH-']
            audio_file_name = os.path.basename(_VARS['audioPath'])
            _VARS['window']['-TABLE-'].update(values=[["Audio",audio_file_name]])
            _VARS['window'].refresh()

        elif event == 'ShowMic':
            threading.Thread(target=runner, args=("python3 plot_input.py", )).start()

        elif event == '-ATTACK-':
            _VARS['window']['-ATTACK-'].update('Generating attacks..')

            audio_path = values['-AUDIO_PATH-'] if values['-AUDIO_PATH-'] else audio_recorder.filepath
            signal, sr = torchaudio.load(audio_path)
            audio_samples_path = os.path.join(_VARS['syntheticSamplesPath'],os.path.basename(audio_path))
            if not os.path.exists(audio_samples_path):
                os.makedirs(audio_samples_path)
            
            label = predict_audio_path(audio_path, deepspeech_model)
            if label == 1:
                sg.popup("The audio is alrealdy Authorized")
                continue


            if audio_path and os.path.exists(audio_path):
                if os.path.exists(audio_samples_path) and len(glob.glob(os.path.join(audio_samples_path, '*.wav'))) > 0:
                    synthetic_samples_dict = load_synthetic_samples(audio_samples_path)
                else:
                    synthetic_samples_dict = generate_synthetic_samples(audio_path, audio_samples_path, deepspeech_model, pytorch_classifier)
                    save_synthetic_samples(synthetic_samples_dict, audio_samples_path)
                
                generated_audio_files = os.listdir(audio_samples_path)
                _VARS['window']['-GENERATED_AUDIOS-'].update(values=generated_audio_files)
                #synthetic_samples_dict = generate_synthetic_samples(audio_path,audio_samples_path,deepspeech_model,pytorch_classifier)
                
                attack_summary = []  # To hold the summary for each model and attack type
                raw_colors = []

                for attack_name, synthetic_samples in synthetic_samples_dict.items():

                    attack_summary.append(["", "", "","",""])
                    raw_colors.append((len(attack_summary)-1, "","","")) 
                    
                    for model_name in values['-MODEL-']:
                        name_to_model = {v: k for k, v in models.items()}
                        model = name_to_model.get(model_name)
                        if model:
                            total_samples = len(synthetic_samples)
                            authorized_count = 0
                            perturbations = []
                            SNRs = []
                            for synthetic_sample in synthetic_samples:
                                perturbation = get_mean_perturbation(signal,synthetic_sample)
                                snr = SNR(signal,synthetic_sample)
                                perturbations.append(perturbation)
                                if snr is not None:
                                    SNRs.append(snr)
                                label = predict_audio(synthetic_sample, model)
                                if label == 1:  # Assuming 1 represents "Authorized"
                                    authorized_count += 1
                            mean_pert = round(np.mean(perturbations),3)
                            mean_snr = round(np.mean(SNRs),3)

                            # For single sample scenario, show "Authorized" or "Not Authorized"


                            authorized_percentage = (authorized_count / total_samples) * 100 if total_samples else 0
                            not_authorized_percentage = 100 - authorized_percentage
                            if total_samples == 1:
                                status = "Authorized" if authorized_count == 1 else "Not Authorized"
                                color = 'green' if authorized_count == 1 else 'red'
                                attack_summary.append([model_name, attack_name, status,mean_pert,mean_snr])
                                raw_colors.append( (len(attack_summary)-1 ,color) )

                            else:
                                attack_summary.append([model_name, attack_name, f"{not_authorized_percentage:.2f}% NA"+" "+f"{authorized_percentage:.2f}% A",mean_pert,mean_snr])

                # Update the table with attack summary
                # Ensure the table is configured to display the necessary columns and use the last column for row coloring if present
                table_element = _VARS['window']['-TABLE_ATTACKS-']
                if len(raw_colors)>1:  # Check if table supports coloring
                    table_element.update(values=attack_summary, row_colors=raw_colors)
                else:
                    table_element.update(values=attack_summary)

            else:
                sg.popup("Please select an audio file first.")
            _VARS['window']['-ATTACK-'].update('ATTACK')
            
        elif event == '-PLAY_SELECTED_AUDIO-':
            selected_file = values['-GENERATED_AUDIOS-'][0] if values['-GENERATED_AUDIOS-'] else None
            if selected_file:
                audio_path = os.path.join(audio_samples_path, selected_file)
                play_audio(audio_path)
            else:
                sg.popup("Please select an attack-generated audio file first.")

        elif event == '-PLAY_AUDIO-':
            audio_path = values['-AUDIO_PATH-']  
            play_audio(audio_path)


    _VARS['window'].close()

if __name__ == "__main__":
    main()
    
