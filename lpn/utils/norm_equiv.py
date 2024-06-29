import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="reflect", blind=True):


        super().__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, padding_mode=padding_mode, bias=False)
        self.blind = blind
    
    
    def affine(self, w):
        """ returns new kernels that encode affine combinations """
        return w.view(self.out_channels, -1).roll(1, 1).view(w.size()) - w + 1 / w[0, ...].numel()
    
    def forward(self, x):
        kernel = self.affine(self.weight) if self.blind else torch.cat((self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]), dim=1)
        padding = tuple(elt for elt in reversed(self.padding) for _ in range(2)) # used to translate padding arg used by Conv module to the ones used by F.pad
        padding_mode = self.padding_mode if self.padding_mode != 'zeros' else 'constant' # used to translate padding_mode arg used by Conv module to the ones used by F.pad
        #print(x.shape)
        #print(padding)
        pad = F.pad(x, padding, mode=padding_mode)
        #print(pad.shape)
        #print(pad)
        return F.conv2d(pad, kernel, stride=self.stride, dilation=self.dilation, groups=self.groups)
    #padding in the input should be a series of ints at least of size 2
    #padding should match convolution dimensions
    #padding_mode should be a string


class SortPool(nn.Module):
    """ Channel-wise sort pooling, C must be an even number """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        #print(x)
        #print(x.size())
        if (len(x.size()) == 4):
            N, C, H, W = x.size()
            x1, x2 = torch.split(x.view(N, C//2, 2, H, W), 1, dim=2)
            diff = F.relu(x1 - x2, inplace=True)
            return torch.cat((x1-diff, x2+diff), dim=2).view(N, C, H, W)
        else:
            N, C = x.size()
            x = x.unsqueeze(-1)
            x = x.unsqueeze(-1)
            H, W = 1, 1
            x1, x2 = torch.split(x.view(N, C//2, 2, H, W), 1, dim=2)
            diff = F.relu(x1 - x2, inplace=True)
            x = torch.cat((x1-diff, x2+diff), dim=2).view(N, C, H, W)
            x = x.squeeze(-1)
            x = x.squeeze(-1)
            return x
