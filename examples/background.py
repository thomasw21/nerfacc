import torch
from torch import nn

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

class MLPBackground(nn.Module):
    def __init__(
        self,
        in_dim: int = 3, # viewing directions
        hidden_dim: int = 64,
        out_dim: int = 3, # RGB colors
        num_layers: int = 2
    ):
        super().__init__()

        self.encoder = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "SphericalHarmonics",
                "n_dims_to_encode": in_dim,
                "degree": 4,
            },
        )
        # self.encoder = tcnn.Encoding(
        #     n_input_dims=in_dim,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 12
        #     },
        # )

        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=out_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    def forward(self, x: torch.Tensor):
        # Spherical harmonics preprocessing
        x = (x + 1.0) / 2.0
        return self.mlp_base(self.encoder(x))

    def get_params(self, lr: float):
        return [
            {"params": list(child.parameters()), "lr": lr * 10 if isinstance(child, tcnn.Encoding) else lr}
            for child in self.children()
        ]
