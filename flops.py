import networks
import torch
import torch.nn as nn
from torchinfo import summary
# model = networks.LBPGenerator()
# print(model)
lbp = torch.randn(1,1, 256, 256)
mask = torch.randn(1,1, 256, 256)
x = torch.randn(1,3, 256, 256)
# summary(model,input_data=(lbp, mask))

modelG =networks.ImageGenerator()
print(modelG)
summary(modelG,input_data=(x,lbp, mask))