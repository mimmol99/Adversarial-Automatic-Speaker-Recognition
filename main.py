from Dataset import SpReWDataset
from Models import ECAPA_TDNN_SpeechBrain
from Evaluation import training_phase,test_phase
from Dataset import EmbeddingDatasetSI
from EmbeddingClassifier import LinearEmbeddingClassifier

import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader

def main():

    #setup
    dataset_path = "./SpReW"
    sprew = SpReWDataset(base_path = dataset_path,generate_signals = False)
    ecapa_tdnn = ECAPA_TDNN_SpeechBrain()

    #split paths
    tr_paths, val_paths, test_paths = sprew.split_paths(split_training=0.7, split_validation=0.25, split_test=0.05)
    train_labels = [sprew.get_label_from_path(path) for path in tr_paths]
    val_labels = [sprew.get_label_from_path(path) for path in val_paths]
    test_labels = [sprew.get_label_from_path(path) for path in test_paths]
    
    #extract embeddings
    tr_embeddings = ecapa_tdnn.get_embedding_from_paths(tr_paths)
    val_embeddings = ecapa_tdnn.get_embedding_from_paths(val_paths)
    test_embeddings = ecapa_tdnn.get_embedding_from_paths(test_paths)

    #Dataloaders
    train_embedding_dataset_si = EmbeddingDatasetSI(tr_embeddings,train_labels)
    eval_embedding_dataset_si = EmbeddingDatasetSI(val_embeddings,val_labels)
    test_embeddings_dataset_si = EmbeddingDatasetSI(test_embeddings,test_labels)

    train_loader_si = DataLoader(train_embedding_dataset_si, batch_size = 8, shuffle=True)
    eval_loader_si = DataLoader(eval_embedding_dataset_si, batch_size = 8)
    test_loader_si = DataLoader(test_embeddings_dataset_si,batch_size = 8)

    #embedding classifier
    embedding_classifier_si = LinearEmbeddingClassifier(embedding_size = 192, num_classes = 20,embedding_model= ecapa_tdnn)

    #hyperparameters setting
    criterion_si = nn.CrossEntropyLoss()
    optimizer_si = optim.SGD(embedding_classifier_si.parameters(), lr=1e-4, momentum=0.9)
    epochs = 3
    patience = epochs // 2

    #train
    embedding_classifier_si_trained = training_phase(embedding_classifier_si,train_loader= train_loader_si,eval_loader= eval_loader_si,criterion=criterion_si,optimizer=optimizer_si,epochs= epochs,patience= patience)
    #test
    test_phase(embedding_classifier_si_trained,test_loader= test_loader_si,criterion= criterion_si)
    
    # Save the model
    torch.save(embedding_classifier_si_trained, 'embedding_classifier_si_trained_model.pt')


if __name__ == "__main__":
    main()
