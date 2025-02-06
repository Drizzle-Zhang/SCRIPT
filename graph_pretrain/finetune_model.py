import torch
from torch import nn
from tqdm import tqdm


class FinetuneModel(nn.Module):
    def __init__(self, pooling_method, projection_method, hidden_size, in_dim, out_dim, loss_fn, device):
        super(FinetuneModel, self).__init__()
        self.pooling_method = pooling_method
        self.projection_method = projection_method
        self.hidden_size = hidden_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.loss_fn = loss_fn
        self.projection_layer = self.get_projection_layer()
        self.loss_criterion = self.get_loss_criterion()
        self.device = device
        self.loss_scalar = 10e2

    def forward(self, inputs, mask):
        pooling_tensor = self.pooling_layer(inputs, mask)
        projection_tensor = self.projection_layer(pooling_tensor)
        return projection_tensor

    def pooling_layer(self, inputs, mask):
        # inputs: [batch, 1574, num_of_node, dim] -> [batch, 1574, dim]
        if self.pooling_method == 'max':
            mask = mask.unsqueeze(-1).expand(inputs.size()).float()
            inputs[mask == 0] = -1e4
            return torch.max(inputs, dim=2).values
        elif self.pooling_method == 'mean':
            mask = mask.unsqueeze(-1).expand(inputs.size()).float()
            return torch.sum(inputs, dim=2) / torch.sum(mask, dim=2) + 1e-6
        else:
            raise NotImplementedError(f'Pooling method {self.pooling_method} not implemented')

    def projection(self, inputs):
        return self.projection_layer(inputs)

    def get_loss(self, out_exp, true_exp):
        if self.loss_fn == "kl_div":
            out_exp = torch.nn.functional.log_softmax(out_exp.squeeze(), dim=1)
            # true_exp = torch.nn.functional.log_softmax(true_exp.squeeze(), dim=1)
            print('-'*10, out_exp, true_exp)
            exp_loss = self.loss_criterion(out_exp, true_exp)
        elif self.loss_fn in ("mse", "mae"):
            exp_loss = self.loss_scalar * self.loss_criterion(out_exp.squeeze(), true_exp.squeeze())
        else:
            exp_loss = self.loss_criterion(out_exp.squeeze(), true_exp.squeeze())
        return exp_loss

    def get_projection_layer(self):
        if self.projection_method == 'mlp':
            return nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.out_dim),
            )
        else:
            raise NotImplementedError(f'Projection method {self.projection_method} not implemented')

    def get_loss_criterion(self):
        if self.loss_fn == 'kl_div':
            return nn.KLDivLoss(log_target=False, reduction='batchmean')
        elif self.loss_fn == 'mse':
            return nn.MSELoss(reduction='mean')
        elif self.loss_fn == 'mae':
            return nn.L1Loss(reduction='mean')
        elif self.loss_fn == 'poisson':
            return nn.PoissonNLLLoss(reduction="mean")
        else:
            raise NotImplementedError(f'Loss function {self.loss_fn} not implemented')
