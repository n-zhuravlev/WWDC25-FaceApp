import coremltools as ct
import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import irfftn, rfftn


class ConvBlock(nn.Module):
    """
    This is a simple convolution block for demonstration purposes.
    The structure is the following:
        Conv -> Act -> Conv -> Act
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FFCBlock(nn.Module):
    """
    This module represents a Fast-Fourier-Transform-based convolution block.
    We use complicated combinations of rfft/irfft operations in our networks, but this is enough for demonstration purposes.
    The structure is the following:
        Conv -> rfftn -> FFN -> irfftn -> Conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Conv2d(2 * out_ch, 2 * out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * out_ch, 2 * out_ch, kernel_size=1),
        )
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = rfftn(x, dim=(-2, -1), norm="ortho")
        x = torch.cat((x.real, x.imag), dim=1)
        x = self.ffn(x)
        x = irfftn(torch.complex(*torch.split(x, x.shape[1] // 2, dim=1)), dim=(-2, -1), norm="ortho")
        x = self.conv2(x)
        return x


class TransformerBlock(nn.Module):
    """
    This is a simple convolution-like transformer block for demonstration purposes.
    The structure is the following:
        Q -> K -> V -> Attention -> Proj -> FFN
    """
    _NUM_HEADS = 8
    _QK_DIM = 32
    _V_DIM = 64

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.q = nn.Conv2d(in_ch, self._NUM_HEADS * self._QK_DIM, kernel_size=1)
        self.k = nn.Conv2d(in_ch, self._NUM_HEADS * self._QK_DIM, kernel_size=1)
        self.v = nn.Conv2d(in_ch, self._NUM_HEADS * self._V_DIM, kernel_size=1)
        self.proj = nn.Conv2d(self._NUM_HEADS * self._V_DIM, out_ch, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(-1, self._NUM_HEADS, self._QK_DIM, H * W).permute(0, 1, 3, 2)
        k = k.view(-1, self._NUM_HEADS, self._QK_DIM, H * W).permute(0, 1, 3, 2)
        v = v.view(-1, self._NUM_HEADS, self._V_DIM, H * W).permute(0, 1, 3, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 1, 3, 2).reshape(-1, self._NUM_HEADS * self._V_DIM, H, W)
        x = self.proj(x)
        x = self.ffn(x)
        return x


class DownsamplingBlock(nn.Module):
    """This is a simple downsampling block for demonstration purposes."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class UpsamplingBlock(nn.Module):
    """This is a simple upsampling block for demonstration purposes."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='nearest')


class DemonstrationModel(nn.Module):
    """
    This is a simple demonstration model that combines convolution, FFC, and transformer blocks.
    The structure is the following:
        Stem -> Encoder -> Decoder -> Head
    """
    @staticmethod
    def _build_conv_blocks(num: int, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build a sequence of convolution blocks and downsample block at the end."""
        layers = []
        for _ in range(num):
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        layers.append(DownsamplingBlock())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_transformer_blocks(num: int, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build a sequence of transformer blocks and downsample block at the end."""
        layers = []
        for _ in range(num):
            layers.append(TransformerBlock(in_ch, out_ch))
            in_ch = out_ch
        layers.append(DownsamplingBlock())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_ffc_blocks(num: int, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build an upsample block and a sequence of FFC blocks."""
        layers = []
        layers.append(UpsamplingBlock())
        for _ in range(num):
            layers.append(FFCBlock(in_ch, out_ch))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
        )
        self.encoder = nn.Sequential(
            self._build_conv_blocks(num=5, in_ch=40, out_ch=64),
            self._build_conv_blocks(num=5, in_ch=64, out_ch=96),
            self._build_transformer_blocks(num=5, in_ch=96, out_ch=128),
            self._build_transformer_blocks(num=10, in_ch=128, out_ch=128),
        )
        self.decoder = nn.Sequential(
            self._build_ffc_blocks(num=2, in_ch=128, out_ch=96),
            self._build_ffc_blocks(num=2, in_ch=96, out_ch=64),
            self._build_ffc_blocks(num=2, in_ch=64, out_ch=40),
        )
        self.head = nn.Sequential(
            UpsamplingBlock(),
            nn.Conv2d(40, 20, kernel_size=3, padding=1),
            UpsamplingBlock(),
            nn.Conv2d(20, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = torch.jit.trace(DemonstrationModel(), example_inputs=[torch.randn(1, 3, 1024, 1024)])
    coreml_model = ct.convert(
        model=model, source="pytorch",
        inputs=[ct.ImageType("in_image", shape=(1, 3, 1024, 1024))],
        outputs=[ct.ImageType("out_image")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    coreml_model.save("/workdir/storage/problem1.mlpackage")
