import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticRefinementModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(SemanticRefinementModule, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_h = nn.Conv2d(inplane, outplane, 1, bias=False)

    def forward(self, x):
        l_f, h_f = x
        _, _, h, w = l_f.size()
        h_f = self.gap(h_f)
        h_f_1x1 = self.conv1x1_h(h_f)
        l_h_add = torch.add(l_f, h_f_1x1)
        l_h_mul = F.sigmoid(l_h_add) * l_f
        return l_h_mul, l_h_add,F.sigmoid(l_h_add), l_f

class SemanticCalibrationModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(SemanticCalibrationModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)\
        # generate grid
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid)
        return output

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class DoubleConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DoubleConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)