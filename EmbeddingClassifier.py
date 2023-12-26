import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LinearEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes,embedding_model,threshold = 0.5):
        super(LinearEmbeddingClassifier, self).__init__()
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.threshold = threshold

        self.fc1 = nn.Linear(self.embedding_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32,num_classes)



    def forward(self, x):
    # If the input tensor is of shape [1, batch_size, embedding_size]
    # We want to reshape it to [batch_size, embedding_size]
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        elif x.dim() > 3:
            # Handle other cases with more than 3 dimensions if necessary
            raise ValueError("Input tensor has too many dimensions")

        if x.size()[-1] !=self.embedding_size:
            #x = self.embedding_model(x[0])
            x = self.embedding_model.get_embedding_from_signal(x)

        #print("start ec",x)
        #print(len(x),x,isinstance(x, list),isinstance(x, np.ndarray))

        if isinstance(x, list):
            x = torch.stack(x)
            #print(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()


        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)

        #print(x)

        #print("end",x)
        return x


    def score(self, x):
        x = self.forward(x)

        if self.num_classes == 1:
            # For binary classification, return the sigmoid output
            return torch.sigmoid(x)

        else:
            # For multi-class, apply softmax
            return F.softmax(x, dim=1)


    def make_decision(self, x):
        scores = self.score(x)

        if self.num_classes == 1:
            # For binary classification, use a threshold (e.g., 0.5)
            decisions = (scores > self.threshold).int()
        else:
            # For multi-class, use argmax
            decisions = torch.argmax(scores, dim=-1)

        return decisions, scores

