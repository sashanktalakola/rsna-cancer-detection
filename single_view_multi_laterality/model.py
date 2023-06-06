import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, backbone, feature_vec_size):
        super(Model, self).__init__()
        self.feature_vec_size = feature_vec_size
        self.eff_feature_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)

        self.classification_network = nn.Sequential(
            nn.Linear(feature_vec_size, 1),
        )

    def forward(self, X):
        X = self.eff_feature_extractor(X)
        return self.classification_network(X)