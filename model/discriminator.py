import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, dis_dims, loss, pack):
        super(Discriminator, self).__init__()
        torch.manual_seed(0)
        self.pack = pack
        self.packdim = input_dim * pack

        layers = self._build_layers(input_dim, dis_dims, loss)
        self.seq = nn.Sequential(*layers)

        self._initialize_weights()

    def _build_layers(self, input_dim, dis_dims, loss):
        dim = input_dim * self.pack
        layers = []
        for item in dis_dims:
            layers.extend([
                nn.utils.spectral_norm(nn.Linear(dim, item)),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            dim = item
        layers.extend([
            nn.utils.spectral_norm(nn.Linear(dim, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(128, 1))
        ])
        if loss == "cross_entropy":
            layers.append(nn.Sigmoid())
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):
        assert input.size(0) % self.pack == 0, "Input batch size must be divisible by pack size"
        return self.seq(input.view(-1, self.packdim))

    def gradient_penalty(self, real_data, fake_data, device, lambda_=10):
        alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        d_interpolates = self(interpolates)
        fake = torch.ones(d_interpolates.size()).to(device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        return gradient_penalty
