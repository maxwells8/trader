import torch
from torch.autograd import Variable

b = Variable(torch.randn(5, 3), requires_grad=True)
for _ in range(10):
    a = torch.randn(1, 3)
    c = torch.mm(b, torch.t(a))
    (c.sum() * 0.001).backward()
    print(b.grad)
