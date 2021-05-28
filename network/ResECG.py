import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d


class BasicBlock(nn.Module):
    def __init__(self, in_planes, num_filters, stride, idx_block, param):
        super(BasicBlock, self).__init__()
        self.shortcut = nn.MaxPool1d(kernel_size=stride)
        self.idx_block = idx_block
        self.flag_zero_pad = idx_block % param.conv_increase_channels_at == 0 and idx_block > 0
        self.num_filters = num_filters
        self.conv_num_skip = param.conv_num_skip
        if not(idx_block==0):
            self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, num_filters, param.kernel_size_conv, stride=stride, padding=7)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.dropout = nn.Dropout(p=param.dropout_rate)
        self.conv2 = nn.Conv1d(num_filters, num_filters, param.kernel_size_conv, stride=1, padding=7)

    def forward(self, x):
        shortcut = self.shortcut(x)
        if(self.flag_zero_pad):
            shortcut = torch.cat([shortcut, torch.zeros_like(shortcut)], dim = 1)
        if not (self.idx_block==0):
            x = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.conv1(x)))
        out = self.dropout(out)
        out = self.conv2(out)
        out += shortcut
        return out


class ResECG(nn.Module):
    def __init__(self, cfg):
        super(ResECG, self).__init__()
        self.conv1 = nn.Conv1d(1, cfg.num_filters_1st_conv, cfg.kernel_size_conv, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(cfg.num_filters_1st_conv)
        self.layers = self._make_layer(cfg)
        final_planes_conv =  self._get_num_filters_at_index(len(cfg.Strides_ResBlock)-1, 
            cfg.num_filters_1st_conv, cfg.conv_increase_channels_at)
        final_width_div = 1
        for _stride in  cfg.Strides_ResBlock:
            final_width_div *= _stride
        final_width = int(8960 / final_width_div)
        self.output_layer = self._add_output_layer(final_planes_conv, cfg.num_classes)

    def _make_layer(self, params):
        layers = []
        in_planes = params.num_filters_1st_conv
        for idx_block, stride in enumerate(params.Strides_ResBlock):
            num_filters = self._get_num_filters_at_index(idx_block, 
                params.num_filters_1st_conv, params.conv_increase_channels_at)
            layers.append(BasicBlock(in_planes, num_filters, stride, idx_block, params))
            in_planes = num_filters
        return nn.Sequential(*layers)

    def _add_output_layer(self, in_features, out_features):
        return nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.ReLU(inplace=True),
                #nn.Linear(in_features, out_features)
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0)
        )

    def _get_num_filters_at_index(self, idx,  num_start_filters, conv_increase_channels_at):
        return 2**int(idx // conv_increase_channels_at) * num_start_filters


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        #out = out.view(out.shape[0], -1)
        out = self.output_layer(out)
        out = out.squeeze(-1)
        return out

if __name__ == "__main__":
    import yaml
    from munch import Munch
    config = yaml.load(open('./cfgs/ResECG.yaml'))
    model = ResECG(Munch(config['model']))
    input = torch.randn((8,1,8960), dtype=torch.float32)
    output = model(input)
    print(output.shape)

