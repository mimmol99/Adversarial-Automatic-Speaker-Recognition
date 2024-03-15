import pickle
from Dataset import *
from Evaluation import *
from Models import *
from Attacks import *
from art.estimators.classification import PyTorchClassifier
from GUI_v3 import *

def main():

    #Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path =os.get_cwd()
    dataset_path = os.path.join(base_path,"SpReW")
    models_path = os.path.join(base_path,"Models_saved")
    task = "SV"#Or "SI"
    chunk_size = 2
    num_classes = 2 if task == "SV" else 20


    
    sprew_path = os.path.join(base_path,"sprew.pkl")
    if not os.path.exists(sprew_path):
        dataset = ChunkedSpReWDataset(base_path = dataset_path,task = task,chunk_size = chunk_size,generate_signals=True)
        with open(sprew_path, 'wb') as file:
            pickle.dump(dataset,file)
    else:
        with open(sprew_path, 'rb') as file:
            dataset = pickle.load(file)
    #dataset.plot_samples_distribution()
    #dataset.plot_speakers_distribution()
    #dataset.plot_category_distribution()

    #training
    tr_paths = dataset.get_splitted_paths()['training']
    #print(tr_paths[0])
    val_paths = dataset.get_splitted_paths()['validation']
    test_paths = dataset.get_splitted_paths()['test']

    splitted_signals = dataset.get_splitted_signals()

    signals_tr = splitted_signals['training']
    signals_val = splitted_signals['validation']
    signals_test = splitted_signals['test']

    labels_tr = [dataset.get_label_from_path(path) for path in tr_paths]
    labels_val = [dataset.get_label_from_path(path) for path in val_paths]
    labels_test = [dataset.get_label_from_path(path) for path in test_paths]

    train_loader  = to_loader(signals_tr,labels_tr)
    eval_loader  = to_loader(signals_val,labels_val)
    test_loader = to_loader(signals_test, labels_test)

    sr = 16000
    n_feature = sr*chunk_size
    deepspeech_model = DeepSpeech(n_feature = n_feature, n_hidden = 512, n_class = num_classes, dropout= 0.35)
    model_name = "DeepSpeech"
    
    batch_size = 32
    criterion = get_criterion(num_classes=num_classes)
    optimizer = get_optimizer(model=deepspeech_model,num_classes=num_classes,lr=1e-5)

    epochs = 200
    patience = min(epochs//5,10)

    model_save_path = os.path.join(models_path,'model_'+model_name)
    model_save_path_ext = os.path.join(models_path,'model_'+model_name+'.pt')
    if not os.path.exists(model_save_path_ext):
        deepspeech_model = torch_training_phase(deepspeech_model, train_loader, eval_loader, epochs, criterion, optimizer, patience, plot=False, save_name = model_save_path)
        state_dict = torch.load(model_save_path_ext)

    else:
        state_dict = torch.load(model_save_path_ext,map_location=device)
        missing_keys, unexpected_keys = deepspeech_model.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained model: ", model_save_path_ext)
        
    #test
    torch_test_phase(deepspeech_model,test_loader= test_loader,criterion= criterion)


    #loading defended model
    detector = DeepSpeech(n_feature = n_feature, n_hidden = 512, n_class = num_classes, dropout= 0.3)
    model_name = "Detector_DeepSpeech"
    model_save_path = os.path.join(models_path,'model_'+model_name+'.pt')

    state_dict = torch.load(model_save_path,map_location=device)
    missing_keys, unexpected_keys = detector.load_state_dict(state_dict, strict=False)
    print("Loaded pre-trained model: ", model_save_path)

    #read microphone audio input

    #attack audio input
    model_to_attack = deepspeech_model
    min_value = -1
    max_value = 1
    shape = len(signals_tr[0])
    epochs = 1
    pytorch_classifier = PyTorchClassifier(
                                    model=model_to_attack,
                                    clip_values=(min_value, max_value),
                                    loss=criterion,
                                    optimizer=optimizer,
                                    input_shape=shape,
                                    nb_classes=num_classes

                                )

    x_tr_tensor = torch.stack(signals_tr+signals_val)
    labels_tr_tensor = torch.Tensor(labels_tr+labels_val)
    x_tr_cpu = x_tr_tensor.cpu()
    labels_tr_cpu = labels_tr_tensor.cpu()

    #Trained on training and val samples
    pytorch_classifier.fit(x_tr_tensor, labels_tr_tensor, batch_size=batch_size, nb_epochs=1, verbose=True)
    
    x_test_tensor = torch.stack(signals_test)
    labels_test_tensor = torch.Tensor(labels_test)
    predictions = pytorch_classifier.predict(x_test_tensor.cpu().numpy())

    # If your model outputs logits, apply softmax to convert to probabilities
    predictions = softmax(torch.from_numpy(predictions), dim=1).numpy()

    # Convert labels_test_tensor to numpy array if it's a tensor
    if isinstance(labels_test_tensor, torch.Tensor):
        labels_test_tensor = labels_test_tensor.numpy()

    ## Ensure labels_test_tensor is of type 'long' for comparison
    #labels_test_tensor = labels_test_tensor.astype(np.long)

    # Calculate accuracy
    accuracy = np.sum(np.argmax(predictions, axis=1) == labels_test_tensor) / len(labels_test_tensor)
    accuracy = round(accuracy, 3)
    #print("Predictions:", np.argmax(predictions, axis=1))
    #print("True labels:", labels_test_tensor)
    print("Accuracy on test clean examples: {}%".format(accuracy))
    #show output of defended and undefended model

    model_to_attack = deepspeech_model

    defended_model = detector



    audio_path  = random.choice(tr_paths)
    label = dataset.get_label_from_path(audio_path)#dataset.get_label_from_signal(audio)
    while label == 1:
        audio_path  = random.choice(tr_paths)
        label = dataset.get_label_from_path(audio_path)#dataset.get_label_from_signal(audio)
    print(f"path: {audio_path} label: {label}")
    audio,sr = torchaudio.load(audio_path)

    attack_sample = torch.stack([audio])
    attack_label = torch.tensor(0)
    if not isinstance(attack_label, torch.Tensor) or len(attack_label.shape) == 0:
        attack_label = torch.tensor([attack_label])

    # If attack_sample is a single sample without a batch dimension, add the batch dimension
    if len(attack_sample.shape) == 1:
        attack_sample = attack_sample.unsqueeze(0)

    range_values = round(max_value - min_value,3)
    steps = 5
    num_steps = steps + 1
    e_step = round(range_values/steps,3)
    max_v = range_values
    min_v = round(max_value/steps,3)


    
    # Initialize attacks
    fgsm_attack = ART_FGSM(pytorch_classifier)
    bim_attack = ART_BIM(pytorch_classifier)
    pgd_attack = ART_PGD(pytorch_classifier)
    df_attack = ART_DF(pytorch_classifier)
    cw_attack = ART_CW(pytorch_classifier)
    

    # Define parameters for each attack (you should expand these based on your needs)
    # Initialize attacks with adjusted parameters

    pgd_params = {
        'eps': np.round(np.linspace(min_v, max_v, num_steps), 3),
        'eps_step': np.round(np.linspace(min_v / steps, min_v, num_steps), 3),
        'max_iter': [5],
        'targeted': [True],
        'verbose':[False],
    }

    df_params = {
        'max_iter': [25],
        'epsilon': [1e-3, 1],
        'nb_grads': [2],
        'batch_size': [1]
    }

    # Generate parameter combinations
    #fgsm_combinations = ART_Attacks.generate_parameters_dicts(fgsm_params)
    #bim_combinations = ART_Attacks.generate_parameters_dicts(bim_params)
    pgd_combinations = ART_Attacks.generate_parameters_dicts(pgd_params)
    df_combinations = ART_Attacks.generate_parameters_dicts(df_params)
    #cw_combinations = ART_Attacks.generate_parameters_dicts(cw_params)

    # Initialize and use the MultipleARTAttacks class

    art_attacks = {
        #'FGSM': fgsm_attack,
        #'BIM': bim_attack,
        'PGD': pgd_attack,
        'DF': df_attack,
        #'CW': cw_attack,
    }
    art_combinations = {
        #'FGSM': fgsm_combinations,
        #'BIM': bim_combinations,
        'PGD': pgd_combinations,
        'DF': df_combinations,
        #'CW': cw_combinations
    }

    device = 'cpu'
    
    multiple_art_attacks = MultipleARTAttacks(model_to_attack, art_attacks, art_combinations, attack_sample, attack_label)
    '''
    multiple_art_attacks.run_attacks(N_top = 3)
    multiple_art_attacks.analyze_results()
    results = multiple_art_attacks.get_results()
    
    device = 'cpu'

    clean_label_undefended_model = torch.argmax(model_to_attack(attack_sample))
    clean_label_defended_model = torch.argmax(defended_model(attack_sample))

    print(f"Prediction model: {clean_label_defended_model}\nPrediction defended model:{clean_label_defended_model}")
    
    
    for attack_name, result in results[model_to_attack].items():

        top_params = result['params']
        top_samples = result['samples']
        for param,samples in zip(top_params,top_samples):

            sample = samples[0]
            sample_tensor = torch.tensor(sample)
            sample_tensor = sample_tensor.to('cpu')

            attacked_model_predict = torch.argmax(model_to_attack(sample_tensor))
            defended_model_predict = torch.argmax(defended_model(sample_tensor))
            
            print(f"model without defense predicted:{attacked_model_predict}\nmodel defended predicted: {defended_model_predict}")
    '''

    root = tk.Tk()
    models = {model_to_attack:"DeepSpeech", defended_model:"Detector"}
    gui = GUI(root, models, multiple_art_attacks)  # Pass the attack generator object
    root.mainloop()


if __name__ == "__main__":
    main()
