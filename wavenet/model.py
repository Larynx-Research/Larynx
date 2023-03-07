## WAVENET !
import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, in_channel, out_channel, skip_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = non_diluted_conv(in_channel, out_channel, kernel_size)
        self.lazy_conv = atrous(in_channel, out_channel, kernel_size, dilation)
        self.conv_sk = nn.Conv1d(in_channel, skip_channels, kernel_size)

    def forward(self, x):
        Y = self.lazy_conv(x)
        ## multiple non-linearities ?? why ??
        Y_tan,Y_sig = torch.tanh(Y), torch.sigmoid(Y)
        Y = Y_tan * Y_sig
        res_out = self.conv1(Y) + x[-Y.size(1):]
        skip_out = self.conv_sk(Y)
        print(res_out.shape, skip_out.shape)
        return res_out, skip_out

## Building a basic stack ! keep it simple, stupid 
class res_stack(nn.Module):
    def __init__(self, stack_size, res_channel, skip_channels, kernel_size):
        super().__init__()
        self.blocks = []
        dilations = self.custom_dilations(stack_size)
        for i in dilations:
            self.res_block = residual_block(res_channel, res_channel, skip_channels, kernel_size, i)
            self.blocks.append(self.res_block)

    def custom_dilations(self, stack_size):
        dilations = [2 ** i for i in range(stack_size)]
        return dilations

    def forward(self, x):
        res_out = x
        skip_out = []
        for block in self.blocks:
            res_out, skip_res = block(res_out)
            skip_out.append(skip_res)
        return res_out, torch.stack(skip_out)


class wavenet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward():
        pass

if __name__ == "__main__":
    x = torch.rand(100).reshape(1,-1,)
    in_channel = 1
    out_channel = 1
    kernel_size = 5
    stack_size=3
    skip = 1
    a = res_stack(stack_size,out_channel, skip, kernel_size)
    a.forward(x)
     
