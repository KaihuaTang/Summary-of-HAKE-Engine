## 2. Wukong大规模中文数据集

因为传统的视觉Transformer的token数相对于文本的token数多很多，但计算attention后大量的视觉token其实对训练贡献很低，因此本文提出了一个基于convolution做token压缩的模块。

模块源代码链接：https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/Noah_WuKong/model/modules.py
```
class TokenReduction(nn.Module):
    def __init__(self, in_channels, num_tokens, num_groups=8, dropout_rate=.0):
        super(TokenReduction, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups
        self.norm = LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.in_channels, kernel_size=(1, 1),
                stride=(1, 1), padding=0, groups=self.num_groups, bias=False),
            nn.Conv2d(
                self.in_channels, self.num_tokens, kernel_size=(1, 1),
                stride=(1, 1), padding=0, bias=False),
        )
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=(1, 1),
            stride=(1, 1), padding=0, groups=self.num_groups, bias=False
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, inputs):
        feature_shape = inputs.shape

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)
        selected = self.attention_maps(selected)
        selected = selected.permute(0, 2, 3, 1)
        selected = selected.contiguous().view(
            feature_shape[0], feature_shape[1] * feature_shape[2], -1)
        selected = selected.permute(0, 2, 1)
        selected = nn.functional.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 3, 1, 2)
        feat = self.feat_conv(feat)
        feat = feat.permute(0, 2, 3, 1)
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)

        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd", selected, feat)
        outputs = self.dropout(outputs)

        return outputs, 
```