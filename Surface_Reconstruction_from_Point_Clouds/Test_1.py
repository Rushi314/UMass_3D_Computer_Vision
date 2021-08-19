import torch
import numpy as np
a  = torch.normal(mean=0.0, std=0.05, size=(10,1))
b  = torch.normal(mean=0.0, std=0.05, size=(1,1))
a  = torch.tensor([])
print(b[0][0].type)
s = np.random.normal(0.0, 0.05, size=1)
print([np.random.normal(0.0, 0.05, 1) for _ in range(10)])