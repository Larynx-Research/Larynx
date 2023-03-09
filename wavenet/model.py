## WAVENET !
import torch
import torch.nn as nn

## Allows one to have larger receptive field with same computation and memory costs while also preserving resolution unless Polling or Strided Convolutions
## https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
class atrous(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation, padding=1):
        super().__init__()
        ## padds for same shape as the input
        self.lazy_conv = nn.Conv1d(in_channel, out_channel, kernel_size, dilation=dilation, bias=False, padding="same")

    def forward(self, x):
        return self.lazy_conv(x)

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
        self.skip_conv = nn.Conv1d(in_channel, skip_channels, kernel_size)

    def forward(self, x):
        Y = self.lazy_conv(x)
        ## multiple non-linearities ?? why ??
        Y_tan,Y_sig = torch.tanh(Y), torch.sigmoid(Y)
        Y = Y_tan * Y_sig
        res_out = self.conv1(Y) + x[-Y.size(1):]
        skip_out = self.skip_conv(Y)
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


## A simple dense net to compute the rest of the network !
class dense_net(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = non_diluted_conv(self.channels, self.channels, kernel_size = 1)

    def forward(self, skips):
        x = torch.mean(skips, dim=0)

        for i in range(2):
            relu = nn.ReLU()
            x = relu(x)
            x = self.conv(x)
            softmax = nn.Softmax(dim=1)
        return softmax(x)


class wavenet(nn.Module):

    #            |----------------------------------------|     *residual*
    #            |                                        |
    #            |    |-- conv -- tanh --|                |
    # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
    #                 |-- conv -- sigm --|     |
    #                                         1x1
    #                                          |
    # ---------------------------------------> + ------------->	*skip*

    def __init__(self, in_channel, out_channel, kernel_size, stack_size):
        super().__init__()
        self.stack_size = stack_size
        self.kernel_size = kernel_size
        self.lazy_conv = atrous(in_channel, out_channel, kernel_size, dilation=1)
        self.res_block_stack = res_stack(self.stack_size, in_channel, out_channel, kernel_size)
        self.dense = dense_net(out_channel)

    def forward(self, x):
        x = self.lazy_conv(x)
        _, skips = self.res_block_stack(x)
        x = self.dense(skips)
        print(x)
        return x

def load_wavenet(data=None):
    """FlowNetS model architecture from the
    Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    """
    in_channel = 1
    out_channel = 1
    kernel_size = 5
    stack_size = 16
    model = wavenet(in_channel, out_channel, kernel_size, stack_size)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

## A very simple Wavenet, might not be according to the paper 
if __name__ == "__main__":
    x = torch.rand(3,100).reshape(1,3,-1,)
    in_channel = 3
    out_channel = 3
    kernel_size = 5
    stack_size=3
    skip = 1
    a = wavenet(in_channel,out_channel, kernel_size, stack_size)
    a.forward(x)
