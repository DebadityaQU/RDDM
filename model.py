import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(h_size, 8, batch_first=True)
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SAWrapper(nn.Module):
    def __init__(self, h_size):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)])
        self.h_size = h_size

    def forward(self, x):
        x = self.sa(x.swapaxes(1, 2))
        return x.swapaxes(2, 1)

# U-Net code adapted from: https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #x = self.conv2(x)

        return x

class SegmentUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        
        x1 = self.up(x1)
        #x = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        #x = self.conv2(x)

        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DiffusionUNet(nn.Module):
    def __init__(self, in_size, channels, device):
        super().__init__()
        self.in_size = in_size
        self.channels = channels
        self.device = device
        
        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 2048 // 2)
        
        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)

        self.sa1_x = SAWrapper(128)
        self.sa2_x = SAWrapper(256)
        self.sa3_x = SAWrapper(512)
        self.sa4_x = SAWrapper(1024)
        self.sa5_x = SAWrapper(1024)
        
        self.outc_x = OutConv(32, channels)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)

    def forward(self, x, c, t, verbose=False, arch_type="FULL"):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """

        t = t.unsqueeze(-1)

        # Level 1
        x1 = self.inc_x(x)
        x1 = x1 * c["down_conditions"][0]
        
        if verbose == True:
            print("x1 shape: ", x1.shape)
        
        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, 256)
        
        if verbose == True:
            print("x2 shape: ", x2.shape)
        
        # Level 2
        x2 = self.sa1_x(x2)
        x2 = x2 * c["down_conditions"][1]
        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, 128)
        
        if verbose == True:
            print("x3 shape: ", x3.shape)

        # Level 3
        x3 = x3 * c["down_conditions"][2]
        x3 = self.sa2_x(x3)
        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, 64)
        
        if verbose == True:
            print("x4 shape: ", x4.shape)
        
        # Level 4
        x4 = self.sa3_x(x4)
        x4 = x4 * c["down_conditions"][3]
        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, 32)
        
        if verbose == True:
            print("x5 shape: ", x5.shape)
        
        # Level 5
        x5 = self.sa4_x(x5)
        x5 = x5 * c["down_conditions"][4]
        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, 16)
        
        if verbose == True:
            print("x6 shape: ", x5.shape)
        
        x6 = self.sa5_x(x6)
        x6 = x6 * c["down_conditions"][5]
        
        # Upward path
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, 32)

        if arch_type == "FULL":
            x = x * c["up_conditions"][0]

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, 64)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][1]

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, 128)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][2]

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, 256)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][3]

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, 512)
        
        if arch_type == "FULL":
            x = x * c["up_conditions"][4]

        output = self.outc_x(x)
    
        return output.view(-1, self.channels, 512)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim], elementwise_affine=True)
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([embed_dim], elementwise_affine=True),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, c):
        x_ln = self.ln(x.permute(0, 2, 1))
        c_ln = self.ln(c.permute(0, 2, 1))
        attention_value, _ = self.cross_attention(x_ln, c_ln, c_ln)
        attention_value = attention_value + x_ln
        attention_value = self.ff_cross(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)

class DiffusionUNetCrossAttention(nn.Module):
    def __init__(self, in_size, channels, device, num_heads=8):
        super().__init__()
        self.in_size = in_size
        self.channels = channels
        self.device = device
        
        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 2048 // 2)
        
        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)
        
        self.cross_attention_down1 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_down2 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_down3 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_down4 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_down5 = CrossAttentionBlock(1024, num_heads)
        self.cross_attention_down6 = CrossAttentionBlock(1024, num_heads)

        self.cross_attention_up1 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_up2 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_up3 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_up4 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_up5 = CrossAttentionBlock(32, num_heads)

        self.outc_x = OutConv(32, channels)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)
    
    def forward(self, x, c, t, verbose=False):
        """
        Model is U-Net with added positional encodings and cross-attention layers.
        """
        t = t.unsqueeze(-1)

        # Level 1
        x1 = self.inc_x(x)
        x1 = self.cross_attention_down1(x1, c["down_conditions"][0])
        
        if verbose == True:
            print("x1 shape: ", x1.shape)
        
        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, x1.shape[-1] // 2)
        x2 = self.cross_attention_down2(x2, c["down_conditions"][1])

        if verbose == True:
            print("x2 shape: ", x2.shape)
        
        # Level 2
        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, x1.shape[-1] // 4)
        x3 = self.cross_attention_down3(x3, c["down_conditions"][2])
        
        if verbose == True:
            print("x3 shape: ", x3.shape)

        # Level 3
        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, x1.shape[-1] // 8)
        x4 = self.cross_attention_down4(x4, c["down_conditions"][3])

        if verbose == True:
            print("x4 shape: ", x4.shape)
        
        # Level 4
        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, x1.shape[-1] // 16)
        x5 = self.cross_attention_down5(x5, c["down_conditions"][4])

        if verbose == True:
            print("x5 shape: ", x5.shape)
        
        # Level 5
        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, x1.shape[-1] // 32)
        x6 = self.cross_attention_down6(x6, c["down_conditions"][5])
        
        if verbose == True:
            print("x6 shape: ", x6.shape)
        
        # Upward path
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, x1.shape[-1] // 16)
        x = self.cross_attention_up1(x, c["up_conditions"][0])

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, x1.shape[-1] // 8)
        x = self.cross_attention_up2(x, c["up_conditions"][1])

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, x1.shape[-1] // 4)
        x = self.cross_attention_up3(x, c["up_conditions"][2])

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, x1.shape[-1] // 2)
        x = self.cross_attention_up4(x, c["up_conditions"][3])

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, x1.shape[-1])
        x = self.cross_attention_up5(x, c["up_conditions"][4])

        output = self.outc_x(x)

        return output.view(-1, self.channels, output.shape[-1])

class ConditionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        
        self.inc_c = DoubleConv(1, 64)
        self.inc_freq = DoubleConv(1, 64)

        self.down1_c = Down(64, 128)
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        self.down4_c = Down(512, 1024)
        self.down5_c = Down(1024, 2048 // 2)
        
        self.up1_c = SegmentUp(1024, 512)
        self.up2_c = SegmentUp(512, 256)
        self.up3_c = SegmentUp(256, 128)
        self.up4_c = SegmentUp(128, 64)
        self.up5_c = SegmentUp(64, 32)

    def forward(self, x, verbose=False):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """

        # Level 1

        d1 = self.inc_c(x)
        d2 = self.down1_c(d1)

        if verbose==True:
            print("d2: ", d2.shape)
        
        d3 = self.down2_c(d2)

        if verbose==True:
            print("d3: ", d3.shape)

        d4 = self.down3_c(d3)

        if verbose==True:
            print("d4: ", d4.shape)

        d5 = self.down4_c(d4)

        if verbose==True:
            print("d5: ", d5.shape)
        
        d6 = self.down5_c(d5)

        if verbose==True:
            print("d6: ", d6.shape)
        
        u1 = self.up1_c(d6)
        
        if verbose==True:
            print("u1: ", u1.shape)
        
        u2 = self.up2_c(u1)
        
        if verbose==True:
            print("u2: ", u2.shape)
        
        u3 = self.up3_c(u2)
        
        if verbose==True:
            print("u3: ", u3.shape)

        u4 = self.up4_c(u3)
        
        if verbose==True:
            print("u4: ", u4.shape)

        u5 = self.up5_c(u4)
        
        if verbose==True:
            print("u5: ", u5.shape)

        return {
            "down_conditions": [d1, d2, d3, d4, d5, d6],
            "up_conditions": [u1, u2, u3, u4, u5],
        }

if __name__ == "__main__":

    device = "cuda:0"

    x = torch.randn(2, 1, 128*30).to(device)
    c = torch.randn(2, 1, 128*30).to(device)
    ts = torch.randint(0, 100, [2]).to(device)

    model = DiffusionUNetCrossAttention(512, 1, device=device).to(device)

    conditions = ConditionNet().to(device)(c)

    print(model(x, conditions, ts, verbose=True).shape)