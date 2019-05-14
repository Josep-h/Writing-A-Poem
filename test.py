import torch
x=torch.rand(3,4)
y=torch.rand(3,4)
z=torch.rand(2,3,4)
print(z)
z[0]=x
print(z)