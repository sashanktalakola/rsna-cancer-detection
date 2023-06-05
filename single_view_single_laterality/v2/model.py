import timm
import torch
import torch.nn as nn

class Model(nn.Module):
    def  __init__(self, backbone, feature_vec_size, pretrained, num_auxiliary_features):
        super(Model, self).__init__()
        self.feature_vec_size = feature_vec_size
        self.feature_extractor = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.classification_network = nn.Sequential(
            nn.Linear(feature_vec_size, 1)
        )
        self.auxiliary_networks = nn.ModuleList([
            nn.Linear(feature_vec_size, i) for i in num_auxiliary_features
        ])

    def forward(self, X):
        X = self.feature_extractor(X)
        out_y = self.classification_network(X)
        out_auxiliary = [auxiliary_network(X) for auxiliary_network in self.auxiliary_networks]

        return out_y, out_auxiliary

    def predict(self, X):
        out_y, out_auxiliary = self.forward(X)
        out_y_prob = torch.sigmoid(out_y)
        out_auxiliary_probs = [torch.softmax(auxiliary_output, dim=-1) for auxiliary_output in out_auxiliary]

        return out_y_prob, out_auxiliary_probs