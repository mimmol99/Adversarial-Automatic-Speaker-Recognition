from Dataset import *
from Models import ECAPA_TDNN_SpeechBrain
from Evaluation import training_phase,test_phase
from Dataset import EmbeddingDatasetSI,SignalsSI
from EmbeddingClassifier import LinearEmbeddingClassifier
import nnet
from nnet.ECAPA_TDNN import ECAPA_TDNN
import torch
import math
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses
from SG_Attacks import SG_FAKEBOB,SG_FGSM

def main():

    #setup dataset
    base_path = "./SpReW"
    normalized_base_path = base_path+'_normalized'
    #sprew = SpReWDataset(base_path = base_path,generate_signals = True)
    chunk_size = 1
    sprew_chunk = ChunkedSpReWDataset(base_path,generate_signals = True, chunk_size=chunk_size)
    dataset = sprew_chunk
    authorized_labels = [dataset.speakers_to_int[speaker] for speaker in dataset.authorized_speakers]
    
    n_mels = 160
    num_classes = 2
    embedding_dim = 256
    #ecapa_tdnn_speech_brain = ECAPA_TDNN(n_mels = 1)
    model = ECAPA_TDNN(n_mels = n_mels,embedding_dim = embedding_dim,num_classes= num_classes)
    #model_name = 'ecapa_tdnn_PunkMale'
    

    #split paths
    tr_paths = dataset.get_chunked_paths()['training']
    val_paths = dataset.get_chunked_paths()['validation']
    test_paths = dataset.get_chunked_paths()['test']

    #extract embeddings
    #tr_embeddings = ecapa_tdnn_speech_brain.get_embedding_from_paths(tr_paths)
    #val_embeddings = ecapa_tdnn_speech_brain.get_embedding_from_paths(val_paths)
    #test_embeddings = ecapa_tdnn_speech_brain.get_embedding_from_paths(test_paths)

    #datasets params
    n_paths = 256
    batch_size = 16

    #DataLoaders
    if num_classes >2:

        train_dataset_si = SignalsSI(tr_paths[:n_paths],dataset,authorized_labels)
        val_dataset_si = SignalsSI(val_paths[:n_paths],dataset,authorized_labels)
        test_dataset_si= SignalsSI(test_paths[:n_paths],dataset,authorized_labels)
        #train_dataset_si = MelSpectogramSI(tr_paths[:n_paths],dataset,authorized_labels)
        #val_dataset_si = MelSpectogramSI(val_paths[:n_paths],dataset,authorized_labels)
        #test_dataset_si= MelSpectogramSI(test_paths[:n_paths],dataset,authorized_labels)


        #dataloaders
        train_loader_si  = DataLoader(train_dataset_si, batch_size=batch_size, shuffle=True)
        eval_loader_si  = DataLoader(val_dataset_si, batch_size=batch_size)
        test_loader_si  = DataLoader(test_dataset_si, batch_size=batch_size)

    else:
        to_db = True
        train_dataset_sv = SignalsSV(paths=tr_paths[:n_paths],sprew= dataset,authorized_labels=authorized_labels)
        val_dataset_sv = SignalsSV(paths=val_paths[:n_paths],sprew= dataset,authorized_labels=authorized_labels)
        test_dataset_sv= SignalsSV(paths=test_paths[:n_paths],sprew= dataset,authorized_labels=authorized_labels)
        #train_dataset_sv = MelSpectogramSV(paths=tr_paths[:n_paths],sprew= dataset,authorized_labels=authorized_labels,n_mels=n_mels,to_db=to_db)
        #val_dataset_sv = MelSpectogramSV(paths=val_paths[:n_paths],sprew=dataset,authorized_labels=authorized_labels,n_mels=n_mels,to_db=to_db)
        #test_dataset_sv= MelSpectogramSV(paths=test_paths[:n_paths],sprew=dataset,authorized_labels=authorized_labels,n_mels=n_mels,to_db=to_db)

    

        train_loader_sv  = DataLoader(train_dataset_sv, batch_size=batch_size, shuffle=True)
        eval_loader_sv  = DataLoader(val_dataset_sv, batch_size=batch_size)
        test_loader_sv  = DataLoader(test_dataset_sv, batch_size=batch_size)


    #train hyperparameters
    epochs = 4
    patience = 2


    if num_classes == 1:

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        train_loader = train_loader_sv
        eval_loader = eval_loader_sv
        test_loader = test_loader_sv   

    elif num_classes == 2:

        criterion = nn.CrossEntropyLoss()

        #AAM SOFTMAX to use if using model as embedding extractor
        #criterion = losses.ArcFaceLoss(num_classes, embedding_dim, margin=28.6, scale=64)
        #AAM SOFTMAX requires an optimizer,pass criterion.parameters() to the optimizer

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        train_loader = train_loader_sv
        eval_loader = eval_loader_sv
        test_loader = test_loader_sv

    elif num_classes == 20:

        criterion = nn.CrossEntropyLoss()

        #AAM SOFTMAX to use if using model as embedding extractor
        #criterion = losses.ArcFaceLoss(num_classes, embedding_dim, margin=28.6, scale=64)
        #AAM SOFTMAX requires an optimizer,pass criterion.parameters() to the optimizer

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        train_loader = train_loader_si
        eval_loader = eval_loader_si
        test_loader = test_loader_si



    model_trained = training_phase(model, train_loader, eval_loader, epochs, criterion,optimizer,patience,plot=False)
    
    #test
    test_phase(model_trained,test_loader= test_loader,criterion= criterion)
        
    # Save the model
    torch.save(model_trained.state_dict(), 'model.pt')

    #Attacks
    #fakebob_params = {'model': [model], 'threshold': [None], 'targeted': [True,False], 'epsilon': [0.002], 'max_iter':[2000], 'max_lr': [0.01], 'verbose': [1]}
    fgsm_params = {'model' : [model] , 'epsilon':[0.004]}
    
    attacks_params = [fgsm_params]
    
    #fakebob_attack = SG_FAKEBOB()
    fgsm_attack = SG_FGSM()

    attacks = [fgsm_attack]

    for i,attack in enumerate(attacks):
        comb_success = attack.attacks_combs(test_loader,attacks_params[i])
        print(attack,comb_success)



if __name__ == "__main__":
    main()
