import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, padding=kernel_size//2, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, padding=kernel_size//2, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size, padding=kernel_size//2, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class FFA(nn.Module):
    def __init__(self, gps=3, blocks=19, conv=nn.Conv2d):
        super().__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size, padding=1, bias=True)]
        self.pre = nn.Sequential(*pre_process)

        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.palayer = PALayer(self.dim)

        self.post = nn.Sequential(
            conv(self.dim, 3, kernel_size, padding=1, bias=True)
        )

    def forward(self, x1):
        x = self.pre(x1)
        g1 = self.g1(x)
        g2 = self.g2(g1)
        g3 = self.g3(g2)

        cat = torch.cat([g1, g2, g3], dim=1)
        ca_out = self.ca(cat)
        w = ca_out.view(-1, self.gps, self.dim)[:, :, :, None, None]
        w = w.reshape(-1, self.dim * self.gps, 1, 1)
        out = cat * w
        out = out.view(-1, self.gps, self.dim, *out.shape[2:])
        out = torch.sum(out, dim=1)
        out = self.palayer(out)
        out = self.post(out)
        out = out + x1
        return out


def load_ffa_model(model_path, gps=3, blocks=19, device='cpu'):
    model = FFA(gps=gps, blocks=blocks)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def convert_ffa_to_onnx(model_path, output_path, gps=3, blocks=19):
    model = load_ffa_model(model_path, gps=gps, blocks=blocks)
    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        dynamo=False
    )
    print(f"Model exported to {output_path}")
