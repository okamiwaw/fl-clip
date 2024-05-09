# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Union, cast
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from medclip import constants

__all__ = [
    "VGG",
    "vgg11",
]



model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
}


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        x = self.classifier(feature)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    # single channel input
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {"A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)



# 加载预训练模型和分词器
model_name = constants.BERT_TYPE
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 定义分类器
class Bert_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Bert_Classifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        # Convolutional layer to process the sequence output of BERT
        # Assuming that the sequence length and number of filters are appropriately selected
        self.conv1d = nn.Conv1d(in_channels=self.bert.config.hidden_size,
                                out_channels=128,  # Number of filters
                                kernel_size=3,  # Size of the kernel
                                padding=1)  # Padding to keep sequence length consistent
        # Adaptive max pooling to reduce dimensionality and focus on most significant features
        self.pool = nn.AdaptiveMaxPool1d(1)
        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state which has the shape [batch_size, sequence_length, hidden_size]
        sequence_output = outputs.last_hidden_state
        # Permute the dimensions to [batch_size, hidden_size, sequence_length] for Conv1D
        sequence_output = sequence_output.permute(0, 2, 1)
        # Apply convolution and pooling
        conv_output = self.conv1d(sequence_output)
        pooled_output = self.pool(conv_output).squeeze(-1)  # Remove the last dimension after pooling
        # Dropout for regularization
        dropped = self.dropout(pooled_output)
        # Fully connected layer
        logits = self.fc(dropped)
        probabilities = self.softmax(logits)

        return probabilities