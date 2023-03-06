## WAVENET !
import torch
import torch.nn as nn

## Allows one to have larger receptive field with same computation and memory costs while also preserving resolution unless Polling or Strided Convolutions
## https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
class atrous(nn.Module):
    def __init__(self, in_channel, put_channel, kernel_size, dilation, padding=1):
        super().__init__()
        ## padds for same shape as the input
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, dilation=dilation, bias=False, padding="same")

    def forward(self, x):
        return self.conv1(x)

## Non Diluted Convolutions just for experimental purposes !
class non_diluted_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, bias=False, padding="same")

    def forward(self, x):
        return self.conv(x)

## normal residual blocks as disc in the paper !
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, skip_conn, kernel_size, dilation):
        super().__init__()
        self.conv1 = non_diluted_conv(in_channel, out_channel, kernel_size)
        self.lazy_conv = atrous(in_channel, out_channel, kernel_size, dilation)

    def forward(self, x):
        Y = self.lazy_conv(x)
        Y_tan,Y_sig = torch.tanh(Y), torch.sigmoid(Y)
        Y = Y_tan * Y_sig
        out1 = self.conv1(Y)
        return out1, out2

if __name__ == "__main__":
    x = torch.rand(100).reshape(1,-1,)
    in_channel = 1
    out_channel = 1
    kernel_size = 5
    dilation=3
    skip = 1
    a = residual_block(in_channel, out_channel, skip, kernel_size, dilation)
    a.forward(x)
     
