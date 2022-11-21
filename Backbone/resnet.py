import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import cv2


class Resnet(nn.Module):
    def __init__(self, backbone='resnet101', coco='', input_extra=0, input_height=1024):
        super(Resnet, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=True)
        del self.encoder.fc, self.encoder.avgpool
        if coco:
            coco_pretrain = getattr(models.segmentation, coco)(pretrained=True).backbone
            self.encoder.load_state_dict(coco_pretrain.state_dict())
            # print('coco:')
        self.out_channels = [256, 256, 256, 256]   ####

        self.feat_heights = [input_height//4//(2**i) for i in range(4)]
        # print('self.feat_heights:', self.feat_heights) ## [256, 128, 64, 32]

        if int(backbone[6:]) < 50:
            self.out_channels = [_//4 for _ in self.out_channels]

        print('input_extra:', input_extra)
        # Patch for extra input channel
        if input_extra > 0:
            ori_conv1 = self.encoder.conv1
            new_conv1 = nn.Conv2d(
                3+input_extra, ori_conv1.out_channels,
                kernel_size=ori_conv1.kernel_size,
                stride=ori_conv1.stride,
                padding=ori_conv1.padding,
                bias=ori_conv1.bias)
            with torch.no_grad():
                for i in range(0, 3+input_extra, 3):
                    n = new_conv1.weight[:, i:i+3].shape[1]
                    new_conv1.weight[:, i:i+n] = ori_conv1.weight[:, :n]
            self.encoder.conv1 = new_conv1

        # Prepare for pre/pose down height filtering
        self.pre_down = None
        self.post_down = None
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        print('resnet_x:', x.size(), self.pre_down, self.post_down)

        if self.pre_down is not None:
            x = self.pre_down(x)

        x = self.encoder.layer1(x);

        if self.post_down is not None:
            x = self.post_down(x)

        features.append(x)  # 1/4

        x = self.encoder.layer2(x)
        y2 = self.conv2(x)
        features.append(y2)  # 1/8

        x = self.encoder.layer3(x)
        y3 = self.conv3(x)
        features.append(y3)  # 1/16

        x = self.encoder.layer4(x)
        y4 = self.conv4(x)
        features.append(y4)  # 1/32
        return features

########################################################################################################################

if __name__ == '__main__':
    device = "cuda"

    Encoder = Resnet().to(device)
    print('Encoder:', Encoder)
    # Encoder_kwargs = {"backbone": "resnet101", "input_height":"1024"}
    # encoder_res = Encoder(**Encoder_kwargs)
    # print('encoder_res:', encoder_res)

    # img_path = "/home/zteng/BEVFormer/data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151607028113.jpg"
    # img = cv2.imread(img_path)
    # print('img:', img.shape)

    img = torch.randn([1,3,512,1024]).to(device)
    ml_feat = Encoder(img)
    print('ml_feat:', ml_feat[0].size(), ml_feat[1].size(), ml_feat[2].size(), ml_feat[3].size())

    # summary(Encoder, (1,3, 512, 1024))