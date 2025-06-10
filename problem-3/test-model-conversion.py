import coremltools as ct
import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import irfftn, rfftn


class SimpleConvBlock(nn.Module):

    """
    This is a simple convolution block for demonstration purposes.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SimpleFFCBlock(nn.Module):

    """
    This is a simple convolution block for demonstration purposes.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Conv2d(2 * out_ch, 2 * out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * out_ch, 2 * out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.post = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre(x)
        out = out.to(dtype=torch.float32)
        out = rfftn(out, dim=(-2, -1), norm="ortho")
        out = torch.cat((out.real, out.imag), dim=1)
        out = self.ffn(out)
        out = out.to(dtype=torch.float32)
        real, imag = torch.split(out, out.shape[1] // 2, dim=1)
        out = irfftn(torch.complex(real, imag), dim=(-2, -1), norm="ortho")
        out = self.post(out)
        return out


class SimpleTransformerBlock(nn.Module):

    """
    This is a simple convolution-like transformer block for demonstration purposes.
    The structure is:
        - Attention layer
        - Feed-forward layer
    """

    _NUM_HEADS = 8
    _QK_DIM = 32
    _V_DIM = 64

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Attention part.
        self.q = nn.Conv2d(in_ch, self._NUM_HEADS * self._QK_DIM, kernel_size=1)
        self.k = nn.Conv2d(in_ch, self._NUM_HEADS * self._QK_DIM, kernel_size=1)
        self.v = nn.Conv2d(in_ch, self._NUM_HEADS * self._V_DIM, kernel_size=1)
        self.proj = nn.Conv2d(self._NUM_HEADS * self._V_DIM, out_ch, kernel_size=1)
        # Feed-forward part.
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention part.
        shape = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(-1, self._NUM_HEADS, self._QK_DIM, shape[-2] * shape[-1])
        k = k.view(-1, self._NUM_HEADS, self._QK_DIM, shape[-2] * shape[-1])
        v = v.view(-1, self._NUM_HEADS, self._V_DIM, shape[-2] * shape[-1])
        out = F.scaled_dot_product_attention(q.permute(0, 1, 3, 2), k.permute(0, 1, 3, 2), v.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).reshape(-1, self._NUM_HEADS * self._V_DIM, shape[2], shape[3])
        out = self.proj(out)
        # Feed-forward part.
        out = self.ffn(out)
        return out


class DownsamplingBlock(nn.Module):
    """This is a simple downsampling block for demonstration purposes."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class UpsamplingBlock(nn.Module):
    """This is a simple upsampling block for demonstration purposes."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='nearest')


class SimpleTransformerModel(nn.Module):

    """This is a simple transformer model for demonstration purposes."""

    @staticmethod
    def _build_conv_blocks(num_blocks: int, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build a sequence of convolution blocks."""
        layers = []
        for _ in range(num_blocks):
            layers.append(SimpleConvBlock(in_ch, out_ch))
            in_ch = out_ch
        return nn.Sequential(*layers)

    @staticmethod
    def _build_ffc_blocks(num_blocks: int, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build a sequence of convolution blocks."""
        layers = []
        for _ in range(num_blocks):
            layers.append(SimpleFFCBlock(in_ch, out_ch))
            in_ch = out_ch
        return nn.Sequential(*layers)

    @staticmethod
    def _build_transformer_blocks(num_blocks: int, in_ch: int, out_ch: int) -> nn.Sequential:
        """Build a sequence of convolution blocks."""
        layers = []
        for _ in range(num_blocks):
            layers.append(SimpleTransformerBlock(in_ch, out_ch))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            # Stem.
            nn.Conv2d(3, 20, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            # Encoder Body.
            self._build_conv_blocks(num_blocks=5, in_ch=40, out_ch=64),
            DownsamplingBlock(),
            self._build_conv_blocks(num_blocks=5, in_ch=64, out_ch=96),
            DownsamplingBlock(),
            self._build_transformer_blocks(num_blocks=5, in_ch=96, out_ch=128),
            DownsamplingBlock(),
            self._build_transformer_blocks(num_blocks=10, in_ch=128, out_ch=128),
            # Decoder body.
            UpsamplingBlock(),
            self._build_ffc_blocks(num_blocks=2, in_ch=128, out_ch=96),
            UpsamplingBlock(),
            self._build_ffc_blocks(num_blocks=2, in_ch=96, out_ch=64),
            UpsamplingBlock(),
            self._build_ffc_blocks(num_blocks=2, in_ch=64, out_ch=40),
            # Head.
            UpsamplingBlock(),
            nn.Conv2d(40, 20, kernel_size=3, padding=1),
            UpsamplingBlock(),
            nn.Conv2d(20, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


if __name__ == '__main__':

    model = torch.jit.trace(SimpleTransformerModel(), example_inputs=[torch.randn(1, 3, 1024, 1024)])
    coreml_model = ct.convert(
        model=model, source="pytorch",
        inputs=[ct.ImageType("input_image", shape=(1, 3, 1024, 1024))],
        outputs=[ct.ImageType("output_image")],
        minimum_deployment_target=ct.target.iOS16, convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    coreml_model.save("/workdir/storage/problem1.mlpackage")
