import torch
import torch.nn as nn

class KKANLayer(nn.Module):
  def __init__(self, input_dim, output_dim, g):
      super(KKANLayer, self).__init__()
      self.d = input_dim
      self.k = output_dim
      self.g = g
      self.kernel = lambda w: torch.exp(-(w**2)/2)
      self.x = nn.Parameter(torch.empty(output_dim, 1, g, input_dim))
      nn.init.normal_(self.x, mean=0.0, std=1)
      self.y = nn.Parameter(torch.empty(output_dim, g, 1, input_dim))
      nn.init.normal_(self.y, mean=0.0, std=1)

    # d = self.d # input_dim
    # k = self.k # output_dim
    # n = len(x) # batch
    # g = self.g # grid

  def forward(self, x):
    # normalize x to [-1, 1] using tanh
    x = torch.tanh(x)
    n = len(x)
    Xm = x.reshape(1, n, 1, self.d).expand(self.k, -1, self.g, -1) - self.x.expand(-1, n, -1, -1) # -1 means not changing the size of that dimension # (n, g, input_dim))

    sample_std = torch.std(self.x)
    sample_IQR = torch.quantile(self.x, 0.75) - torch.quantile(self.x, 0.25)
    sigma = min(sample_std, sample_IQR/1.34)
    h = 1.06 * sigma / torch.pow(torch.tensor(float(g)), torch.tensor(1/5))
    _h = 1/h
    Xm = _h * Xm

    Xmk = self.kernel(Xm)
    Xmks = torch.einsum('kngd->knd', Xmk) # sum of g dim
    Xmky = torch.einsum('kngd, kgjd -> knjd', Xmk, self.y) # product
    Xmky = Xmky.view(self.k, n, self.d)
    Xmksi = torch.pow(Xmks, torch.tensor(-1))
    m = torch.einsum('knd,knd->nk', Xmksi, Xmky) # Element-wise product of corresponding rows and sum of d dim (KA theorem) then reshape from k*n to n*k

    return m
