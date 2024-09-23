import torch
from sympy.codegen.scipy_nodes import powm1
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


### This model is similar to EfficientDet but simplified and customized
### For instance, instead of Bi-FPN, here I have implemented custom FPN
class DetectionHead(nn.Module):
    def __init__(self, out_ch, num_classes, num_anchors):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_head = nn.Conv2d(out_ch, num_anchors * num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(out_ch, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, ch, h, w = x.shape
        cls = self.cls_head(x)
        cls = cls.view(batch_size, self.num_anchors, self.num_classes, h, w).permute(0, 3, 4, 1, 2)
        reg = self.reg_head(x)
        reg = reg.view(batch_size, self.num_anchors, 4, h, w).permute(0, 3, 4, 1, 2)
        return cls, reg


class SimplifiedEfficientDet(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(SimplifiedEfficientDet, self).__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        ### Here feature layers are divided so that the image will be divided by two each time
        self.backbone1 = nn.Sequential(backbone.features[:4])  # Output size: [batch, 40, h/8, w/8]
        self.backbone2 = nn.Sequential(backbone.features[4:6])  # Output size: [batch, 112, h/16, w/16]
        self.backbone3 = nn.Sequential(backbone.features[6:8])  # Output size: [batch, 320, h/32, w/32]

        common_feat_ch = 256
        # Lateral 1x1 convolutions to reduce the channels of each feature map to out_channels
        self.channels_matcher1 = nn.Conv2d(40, common_feat_ch, kernel_size=1)
        self.channels_matcher2 = nn.Conv2d(112, common_feat_ch, kernel_size=1)
        self.channels_matcher3 = nn.Conv2d(320, common_feat_ch, kernel_size=1)

        out_ch = 256
        # 3x3 convolutions to smooth the output feature maps after merging
        self.smooth1 = nn.Conv2d(common_feat_ch, out_ch, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(common_feat_ch, out_ch, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(common_feat_ch, out_ch, kernel_size=3, padding=1)

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

        self.detection_head = DetectionHead(out_ch, num_classes, num_anchors)

    def forward(self, x):
        # We pass through efficient net backbone
        feats1 = self.backbone1(x)  # Output size: [batch, 40, h/8, w/8]
        feats2 = self.backbone2(feats1)  # Output size: [batch, 112, h/16, w/16]
        feats3 = self.backbone3(feats2)  # Output size: [batch, 320, h/32, w/32]

        feats_ch_scaled1 = self.channels_matcher1(feats1)  # Output size: [batch, 512, h/8, w/8]
        feats_ch_scaled2 = self.channels_matcher2(feats2)  # Output size: [batch, 512, h/16, w/16]
        feats_ch_scaled3 = self.channels_matcher3(feats3)  # Output size: [batch, 512, h/32, w/32]

        p3 = feats_ch_scaled3  # Output size: [batch, 512, h/32, w/32]
        p2 = feats_ch_scaled2 + self.up_sample(p3)  # Output size: [batch, 512, h/16, w/16]
        p1 = feats_ch_scaled1 + self.up_sample(p2)  # Output size: [batch, 512, h/8, w/8]

        p1 = self.smooth1(p1)
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)

        cls1, reg1 = self.detection_head(p1)
        cls2, reg2 = self.detection_head(p2)
        cls3, reg3 = self.detection_head(p3)

        return [cls1, cls2, cls3], [reg1, reg2, reg3]


class EfficientDetLossFn(nn.Module):
    def __init__(self):
        super(EfficientDetLossFn, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, cnn_cls, cnn_bbox, tgt_cnn_cls, tgt_cnn_bbox, alpha=0.25, gamma=2.0, eps=1e-8):
        loss = 0
        for src_cls, src_bbox, tgt_cls, tgt_bbox in zip(cnn_cls, cnn_bbox, tgt_cnn_cls, tgt_cnn_bbox):
            # Convert logits to probabilities using softmax
            probs = torch.softmax(src_cls, dim=-1)
            # Gather the predicted probabilities for the target classes
            pt = probs[tgt_cls == 1].clamp(min=eps)

            # Compute the focal loss
            focal_loss_value = -alpha * ((1 - pt) ** gamma) * pt.log()

            foreground_mask = (tgt_cls[:, :, :, :, 0] != 1).unsqueeze(-1).float()
            # Regression loss (Smooth L1 or L2 loss)
            regression_loss = self.l1_loss(src_bbox, tgt_bbox) * foreground_mask

            loss += focal_loss_value.mean() + regression_loss.mean()

        return loss


if __name__ == '__main__':
    net = SimplifiedEfficientDet(81, 9)
    inp = torch.randn((10, 3, 64, 64))

    src_cls, src_bbox = net(inp)
    print(src_cls[0].shape, src_bbox[0].shape)
    print(src_cls[1].shape, src_bbox[1].shape)
    print(src_cls[2].shape, src_bbox[2].shape)
