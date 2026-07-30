import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d layer with max-norm constraint on weight parameters (standard in EEG DL).
    """
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces
    (Lawhern et al., Journal of Neural Engineering, 2018)
    """
    def __init__(self, in_channels=22, time_steps=1000, n_classes=4, F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        super(EEGNet, self).__init__()

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.n_classes = n_classes

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dWithConstraint(
            F1, F1 * D, (in_channels, 1), groups=F1, bias=False, max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Calculate linear classifier input feature dimension
        out_time = time_steps // (4 * 8)
        self.feature_dim = F2 * out_time

        # Dense Classifier
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def extract_features(self, x):
        """
        Extracts 1D latent bottleneck embeddings for t-SNE visualization.
        Input x shape: (B, C, T) or (B, 1, C, T)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        features = x.flatten(start_dim=1)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolutional Block (captures diverse frequency-temporal receptive fields).
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[15, 31, 63]):
        super(MultiScaleTemporalConv, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ELU()
            ) for k in kernel_sizes
        ])
        self.fuse_conv = nn.Conv2d(out_channels * len(kernel_sizes), out_channels, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=1)
        out = self.act(self.bn(self.fuse_conv(concat)))
        return out


class TCNBlock(nn.Module):
    """
    Dilated Temporal Convolutional Block for sequential feature extraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.3):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))

        return F.elu(out + residual)


class EEGConformerATCNet(nn.Module):
    """
    SOTA Hybrid Model: Multi-branch Spatial-Temporal Conformer / ATCNet Variant.
    Features:
    1. Multi-scale temporal convolutions + 22-channel depthwise spatial convs.
    2. Multi-Head Self-Attention (MHSA) Transformer block for temporal dynamics.
    3. Dilated Temporal Convolutional Network (TCN) blocks.
    4. Dynamic Gated Feature Fusion layer connecting convolutional and attention embeddings.
    """
    def __init__(
        self,
        in_channels=22,
        time_steps=1000,
        n_classes=4,
        conv_filters=32,
        multi_scale_kernels=[15, 31, 63],
        spatial_depth=2,
        transformer_heads=4,
        transformer_dim=64,
        tcn_channels=[32, 64],
        tcn_kernel_size=5,
        dropout=0.3
    ):
        super(EEGConformerATCNet, self).__init__()

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.n_classes = n_classes

        # 1. Multi-scale Temporal Conv
        self.multi_scale_conv = MultiScaleTemporalConv(
            in_channels=1, out_channels=conv_filters, kernel_sizes=multi_scale_kernels
        )

        # 2. Depthwise Spatial Conv across EEG channels
        out_spatial = conv_filters * spatial_depth
        self.spatial_conv = Conv2dWithConstraint(
            conv_filters, out_spatial, (in_channels, 1),
            groups=conv_filters, bias=False, max_norm=1.0
        )
        self.bn_spatial = nn.BatchNorm2d(out_spatial)
        self.act_spatial = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Temporal sub-sampling dimension
        reduced_time = time_steps // 4

        # Projection to Transformer embedding dimension
        self.proj_transformer = nn.Linear(out_spatial, transformer_dim)
        
        # 3. Multi-Head Self-Attention Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 2, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. Dilated TCN Branch
        self.tcn_block1 = TCNBlock(out_spatial, tcn_channels[0], kernel_size=tcn_kernel_size, dilation=1, dropout=dropout)
        self.tcn_block2 = TCNBlock(tcn_channels[0], tcn_channels[1], kernel_size=tcn_kernel_size, dilation=2, dropout=dropout)

        # Pooling to aggregate temporal features
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 5. Dynamic Gated Feature Fusion
        self.conv_feat_dim = tcn_channels[1]
        self.trans_feat_dim = transformer_dim

        self.gate = nn.Sequential(
            nn.Linear(self.conv_feat_dim + self.trans_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

        # Feature bottleneck dimension
        self.bottleneck_dim = 64
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.conv_feat_dim + self.trans_feat_dim, self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Final Classifier Head
        self.classifier = nn.Linear(self.bottleneck_dim, n_classes)

    def extract_features(self, x):
        """
        Extracts multi-branch fused bottleneck representation.
        x shape: (B, C, T) or (B, 1, C, T)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        # Spatial-Temporal Convolutions
        x_ms = self.multi_scale_conv(x)  # (B, conv_filters, C, T)
        x_sp = self.spatial_conv(x_ms)   # (B, out_spatial, 1, T)
        x_sp = self.act_spatial(self.bn_spatial(x_sp))
        x_sp = self.pool1(x_sp)          # (B, out_spatial, 1, reduced_time)
        x_sp = self.drop1(x_sp)

        # Format for branches
        # Spatial conv features for TCN: (B, out_spatial, reduced_time)
        x_tcn_in = x_sp.squeeze(2)

        # 1. TCN Branch
        x_tcn = self.tcn_block1(x_tcn_in)
        x_tcn = self.tcn_block2(x_tcn)
        tcn_feat = self.global_pool(x_tcn).squeeze(-1)  # (B, tcn_channels[1])

        # 2. Transformer Branch
        # Permute for sequence format: (B, reduced_time, out_spatial)
        x_trans_in = x_sp.squeeze(2).permute(0, 2, 1)
        x_trans_in = self.proj_transformer(x_trans_in)  # (B, reduced_time, transformer_dim)
        x_trans = self.transformer_encoder(x_trans_in)
        trans_feat = x_trans.mean(dim=1)                # (B, transformer_dim)

        # 3. Dynamic Gated Feature Fusion
        combined = torch.cat([tcn_feat, trans_feat], dim=-1)
        weights = self.gate(combined)  # (B, 2)
        
        weighted_tcn = tcn_feat * weights[:, 0:1]
        weighted_trans = trans_feat * weights[:, 1:2]
        
        fused = torch.cat([weighted_tcn, weighted_trans], dim=-1)
        bottleneck = self.fusion_proj(fused)  # (B, bottleneck_dim)

        return bottleneck

    def forward(self, x):
        bottleneck = self.extract_features(x)
        logits = self.classifier(bottleneck)
        return logits


def get_model(config):
    """
    Factory function to instantiate the selected model architecture.
    """
    model_name = config['models'].get('active_model', 'ShallowFBCSPNet')
    in_channels = config['dataset']['num_channels']
    time_steps = config['dataset']['time_steps']
    n_classes = config['dataset']['num_classes']

    if model_name == 'ShallowFBCSPNet':
        m_cfg = config['models']['ShallowFBCSPNet']
        model = ShallowFBCSPNet(
            in_channels=in_channels,
            time_steps=time_steps,
            n_classes=n_classes,
            n_filters=m_cfg.get('n_filters', 40),
            filter_time_length=m_cfg.get('filter_time_length', 25),
            pool_time_length=m_cfg.get('pool_time_length', 75),
            pool_time_stride=m_cfg.get('pool_time_stride', 15),
            dropout=m_cfg.get('dropout', 0.5)
        )
    elif model_name == 'EEGNet':
        m_cfg = config['models']['EEGNet']
        model = EEGNet(
            in_channels=in_channels,
            time_steps=time_steps,
            n_classes=n_classes,
            F1=m_cfg['F1'],
            D=m_cfg['D'],
            F2=m_cfg['F2'],
            kernel_length=m_cfg['kernel_length'],
            dropout=m_cfg['dropout']
        )
    elif model_name == 'EEGConformerATCNet':
        m_cfg = config['models']['EEGConformerATCNet']
        model = EEGConformerATCNet(
            in_channels=in_channels,
            time_steps=time_steps,
            n_classes=n_classes,
            conv_filters=m_cfg['conv_filters'],
            multi_scale_kernels=m_cfg['multi_scale_kernels'],
            spatial_depth=m_cfg['spatial_depth'],
            transformer_heads=m_cfg['transformer_heads'],
            transformer_dim=m_cfg['transformer_dim'],
            tcn_channels=m_cfg['tcn_channels'],
            tcn_kernel_size=m_cfg['tcn_kernel_size'],
            dropout=m_cfg['dropout']
        )
    elif model_name == 'CTNet':
        m_cfg = config['models'].get('CTNet', {})
        model = CTNet(
            in_channels=in_channels,
            time_steps=time_steps,
            n_classes=n_classes,
            F1=m_cfg.get('F1', 16),
            D=m_cfg.get('D', 2),
            F2=m_cfg.get('F2', 32),
            kernel_length=m_cfg.get('kernel_length', 64),
            d_model=m_cfg.get('d_model', 64),
            nhead=m_cfg.get('nhead', 4),
            num_layers=m_cfg.get('num_layers', 2),
            dropout=m_cfg.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return model


class PositionalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class CTNet(nn.Module):
    """
    Official CTNet Architecture from snailpt/CTNet (Nature Scientific Reports 2024):
    https://github.com/snailpt/CTNet
    Features:
    - EEGNet-inspired Convolutional Patch Embedding (Temporal + Spatial Depthwise Conv + 2x AvgPool)
    - Positional Encoding & Multi-Head Attention Transformer Encoder
    - Residual Connection between CNN Patches and Transformer Outputs (features = cnn + trans)
    """
    def __init__(
        self,
        in_channels=22,
        time_steps=1000,
        n_classes=4,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        d_model=16,
        nhead=2,
        num_layers=6,
        dropout=0.3
    ):
        super(CTNet, self).__init__()

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.n_classes = n_classes
        self.F1 = F1
        self.F2 = F1 * D
        self.d_model = d_model

        # 1. Temporal Convolution (0.25s kernel at 250Hz = 64)
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # 2. Channel Depthwise Spatial Conv
        self.depthwise = Conv2dWithConstraint(
            F1, self.F2, (in_channels, 1), groups=F1, bias=False, max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(dropout)

        # 3. Spatial Pointwise / Separable Conv
        self.conv_spatial = nn.Conv2d(self.F2, self.F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Dimension projection to d_model for Transformer
        self.proj = nn.Linear(self.F2, d_model) if self.F2 != d_model else nn.Identity()

        # Positional Encoding & Transformer Encoder Block
        self.positional_encoding = PositionalEncoding(d_model, length=100, dropout=0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Head
        self.reduced_time = time_steps // (8 * 8)
        self.flatten_dim = self.reduced_time * d_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_dim, n_classes)
        )

    def extract_features(self, x):
        """
        Extract features using official CTNet CNN + Transformer + Residual Skip Connection pipeline.
        Input x shape: (B, C, T) or (B, 1, C, T)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_spatial(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Reshape to sequence: (B, reduced_time, F2)
        cnn_seq = x.squeeze(2).permute(0, 2, 1)
        cnn_seq = self.proj(cnn_seq)

        # Positional Encoding & Transformer
        cnn_pe = cnn_seq * (self.d_model ** 0.5)
        cnn_pe = self.positional_encoding(cnn_pe)
        trans_out = self.transformer_encoder(cnn_pe)

        # Official CTNet Residual Skip Connection: features = cnn + trans
        features = cnn_pe + trans_out
        return features

    def forward(self, x):
        features = self.extract_features(x)
        flattened = features.flatten(start_dim=1)
        logits = self.classifier(flattened)
        return logits


class ShallowFBCSPNet(nn.Module):
    """
    Shallow ConvNet / ShallowFBCSPNet for EEG-based Motor Imagery Classification
    (Schirrmeister et al., Human Brain Mapping, 2017)
    Specially engineered with squared activation, temporal pooling, and log transformation.
    """
    def __init__(
        self,
        in_channels=22,
        time_steps=1000,
        n_classes=4,
        n_filters=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        dropout=0.5
    ):
        super(ShallowFBCSPNet, self).__init__()

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.n_classes = n_classes

        # 1. Temporal Convolution (Frequency Bandpass Filters)
        self.conv_time = nn.Conv2d(
            1, n_filters, (1, filter_time_length),
            padding=(0, filter_time_length // 2), bias=False
        )

        # 2. Spatial Depthwise Convolution (Channels) with Max-Norm Constraint
        self.conv_spat = Conv2dWithConstraint(
            n_filters, n_filters, (in_channels, 1),
            groups=n_filters, bias=False, max_norm=2.0
        )
        self.bn = nn.BatchNorm2d(n_filters)
        
        # 3. Log-Power Pooling (x^2 -> AvgPool -> log(x))
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)

        # Calculate temporal output shape after pooling
        pooled_time = (time_steps - pool_time_length) // pool_time_stride + 1
        self.feature_dim = n_filters * pooled_time

        # Classifier Linear Layer
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def extract_features(self, x):
        """
        Extract bottleneck representations for t-SNE visualization.
        Input x shape: (B, C, T) or (B, 1, C, T)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        
        # Log-Power activation & pooling: AvgPool(x^2) -> log
        x = self.pool(x ** 2)
        x = torch.log(torch.clamp(x, min=1e-5))
        x = self.drop(x)

        features = x.flatten(start_dim=1)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

