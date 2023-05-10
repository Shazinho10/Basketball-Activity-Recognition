import torch
from torch import Tensor
from torch.nn import functional as F
from sf_net.semseg.models.base import BaseModel
from sf_net.semseg.models.heads import SFHead


class SFNet(BaseModel):
    def __init__(self, backbone: str = 'ResNetD-18', num_classes: int = 19):
        assert 'ResNet' in backbone
        super().__init__(backbone, num_classes)
        self.head = SFHead(self.backbone.channels, 128 if '18' in backbone else 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        outs = self.backbone(x)
        out = self.head(outs)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out


if __name__ == '__main__':
    model = SFNet('ResNetD-50')
    #model.init_pretrained('checkpoints/backbones/resnetd/resnetd50.pth')
    model.init_pretrained('resnetd50.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
