import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, backbone, feature_vec_size, pretrained):
        super(Model, self).__init__()
        self.feature_vec_size = feature_vec_size
        self.eff_feature_extractor = timm.create_model(backbone, pretrained=pretrained, num_classes=1)

    def forward(self, X):
        out = self.eff_feature_extractor(X)
        return out