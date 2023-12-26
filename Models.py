import speechbrain
from speechbrain.pretrained import SpeakerRecognition
import numpy as np
import torch
from torch import nn, optim
import torchaudio

class ECAPA_TDNN_SpeechBrain():

    def __init__(self):
        self.model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")
        self.path_to_embeddings = {}
        self.signal_to_embeddings = {}

    def get_embedding_from_signal(self,signal):
        if signal in self.signal_to_embeddings.keys():
            return self.signal_to_embeddings[signal]
        else:
            embedding = self.model.encode_batch(signal)
            embedding = embedding[0, 0].detach().numpy()
            self.signal_to_embeddings[signal] = embedding
            return embedding
        
    def get_embedding_from_path(self,path):
        if path in self.path_to_embeddings.keys():
            return self.path_to_embeddings[path]
        else:
            signal, fs = torchaudio.load(path)
            embedding = self.get_embedding_from_signal(signal)
            self.path_to_embeddings[path] = embedding
            return embedding
        
    def get_embedding_from_paths(self,paths):
        embeddings = []
        for path in paths:
           embeddings.append(self.get_embedding_from_path(path))
        return embeddings
         

class UnifiedModel(nn.Module):

    def __init__(self, embedding_model, embedding_classifier):
        super(UnifiedModel, self).__init__()
        self.embedding_model = embedding_model
        self.embedding_classifier = embedding_classifier

    

    def embedding(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Assuming x is a batch of data (either paths or embeddings)

        if isinstance(x, str) and os.path.isfile(x):
            return self.process_single_item(x)

        if x.ndim == 2:
            # Process each item in the batch
            embeddings = [self.process_single_item(item) for item in x]
            return torch.stack(embeddings)
        else:
            return self.process_single_item(x)

    def process_single_item(self, item):

        if isinstance(item, str) and os.path.isfile(item):
            item = get_embedding_from_path(self.embedding_model, item)

        if isinstance(item,np.ndarray):
            item = torch.from_numpy(item)

        #if item.ndim == 1:
        #   item = item.unsqueeze(0)

        return item

    def forward(self, x):

        embedding = self.embedding(x)
        output = self.embedding_classifier(embedding)
        return output


    def score(self, x, softmax=True):
        x = self.forward(x)

        # Check if x has at least two dimensions; if not, add a dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if softmax:
            x = nn.functional.softmax(x, dim=1)
        return x


    def make_decision(self, x, softmax=True):

        scores = self.score(x, softmax)
        decisions = torch.argmax(scores, dim=-1)
        # Apply transformation only if it's needed
        if scores.dim() == 3 and scores.shape[1] == 1:
            scores = scores.squeeze(1)

        if isinstance(scores,np.ndarray):
            scores = torch.from_numpy(scores)
        if isinstance(decisions,np.ndarray):
            decisions = torch.from_numpy(decisions)

        return decisions, scores


    def score(self, x):

        x = self.forward(x)
        #print(x.size())
        # Check if x has at least two dimensions; if not, add a dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self.embedding_classifier.num_classes == 1:
            # For binary classification, return the sigmoid output
            x = torch.sigmoid(x)

        else:
            # For multi-class, apply softmax
            x = F.softmax(x, dim=1)

        #print(x.size())
        return x


    def make_decision(self, x):
        scores = self.score(x)

        if self.embedding_classifier.num_classes == 1:
            # For binary classification, use a threshold (e.g., 0.5)
            decisions = (scores > self.embedding_classifier.threshold).int()
        else:
            # For multi-class, use argmax
            decisions = torch.argmax(scores, dim=-1)

        # Apply transformation only if it's needed
        if scores.dim() == 3 and scores.shape[1] == 1:
            scores = scores.squeeze(1)

        if isinstance(scores,np.ndarray):
            scores = torch.from_numpy(scores)
        if isinstance(decisions,np.ndarray):
            decisions = torch.from_numpy(decisions)

        return decisions, scores
