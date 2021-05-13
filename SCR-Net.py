from modules import *

class SCR_Net(nn.Module):
    def __init__(self, n_channel=3, n_class=1):
        super(SCR_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        filters = [16, 32, 64, 128, 8] # The number of channels can be adjusted

        self.Conv1 = DoubleConvLayer(ch_in=n_channel, ch_out=filters[0])
        self.Conv2 = DoubleConvLayer(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = DoubleConvLayer(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = DoubleConvLayer(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = DoubleConvLayer(ch_in=filters[3], ch_out=filters[3])

        # self.PPM = PSPModule(features=filters[3], out_features=filters[3])
        self.SCM1 = SemanticCalibrationModule(filters[3], filters[3])

        self.Conv6 = DoubleConvLayer(ch_in=filters[3], ch_out=filters[2])
        self.SRM1 = SemanticRefinementModule(inplane=filters[3], outplane=filters[2])
        self.DeConv1x1_1 = nn.Conv2d(filters[2], n_class, kernel_size=1, stride=1, padding=0)
        self.up8s = nn.Upsample(scale_factor=8)
        self.Conv1x1_2 = nn.Conv2d(filters[3], filters[2], kernel_size=1, stride=1, padding=0)
        self.SCM2 = SemanticCalibrationModule(filters[2], filters[2])

        self.Conv7 = DoubleConvLayer(ch_in=filters[2], ch_out=filters[1])
        self.SRM2 = SemanticRefinementModule(inplane=filters[2], outplane=filters[1])
        self.DeConv1x1_2 = nn.Conv2d(filters[1], n_class, kernel_size=1, stride=1, padding=0)
        self.up4s = nn.Upsample(scale_factor=4)
        self.Conv1x1_3 = nn.Conv2d(filters[2], filters[1], kernel_size=1, stride=1, padding=0)
        self.SCM3 = SemanticCalibrationModule(filters[1], filters[1])

        self.Conv8 = DoubleConvLayer(ch_in=filters[1], ch_out=filters[0])
        self.SRM3 = SemanticRefinementModule(inplane=filters[1], outplane=filters[0])
        self.DeConv1x1_3 = nn.Conv2d(filters[0], n_class, kernel_size=1, stride=1, padding=0)
        self.up2s = nn.Upsample(scale_factor=2)
        self.Conv1x1_4 = nn.Conv2d(filters[1], filters[0], kernel_size=1, stride=1, padding=0)
        self.SCM4 = SemanticCalibrationModule(filters[0], filters[0])

        self.Conv9 = DoubleConvLayer(ch_in=filters[0], ch_out=filters[4])
        self.SRM4 = SemanticRefinementModule(inplane=filters[0], outplane=filters[4])
        self.DeConv1x1_4 = nn.Conv2d(filters[4], n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        encoderx1 = self.Conv1(x)
        x = self.Maxpool(encoderx1)
        encoderx2 = self.Conv2(x)
        x = self.Maxpool(encoderx2)
        encoderx3 = self.Conv3(x)
        x = self.Maxpool(encoderx3)
        encoderx4 = self.Conv4(x)
        x = self.Maxpool(encoderx4)
        encoderx5 = self.Conv5(x)
        # encoderx5 = self.PPM(encoderx5)

        # decoding path
        decoderx4 = self.Conv6(self.SCM1([encoderx4,encoderx5]) + encoderx4)
        decoderx3 = self.Conv7(self.SCM2([encoderx3,decoderx4]) + encoderx3)
        decoderx2 = self.Conv8(self.SCM3([encoderx2,decoderx3]) + encoderx2)
        decoderx1 = self.Conv9(self.SCM4([encoderx1,decoderx2]) + encoderx1)
        #
        decoderx4_SR = self.SRM1([decoderx4, encoderx5])
        decoderx4_SR = self.DeConv1x1_1(decoderx4_SR)

        decoderx3_SR = self.SRM2([decoderx3, decoderx4])
        decoderx3_SR = self.DeConv1x1_2(decoderx3_SR)

        decoderx2_SR = self.SRM3([decoderx2, decoderx3])
        decoderx2_SR = self.DeConv1x1_3(decoderx2_SR)

        decoderx1_SR = self.SRM4([decoderx1, decoderx2])
        decoderx1_SR = self.DeConv1x1_4(decoderx1_SR)

        output = self.up8s(decoderx4_SR)+self.up4s(decoderx3_SR)+self.up2s(decoderx2_SR)+decoderx1_SR
        output = torch.nn.functional.sigmoid(output)

        return output

