from model.contextunet import ContextUNETR
import torch

class argss:
    def __init__(self, img_size, in_channels, out_channels, context, align_score, n_prompts=1, textencoder=None):
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context = context
        self.align_score = align_score
        self.n_prompts = n_prompts
        self.textencoder = textencoder

args = argss((256,256,96), 1, 1, False, False)
model = ContextUNETR(img_size=(256,256,96), in_channels=1, out_channels=1, args=args).to('cuda')
print(model)
inp = torch.randn(1, 1, 256, 256, 96).to('cuda')
out = model(inp)
print(out.shape)