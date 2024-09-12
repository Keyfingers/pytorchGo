import torch
import numpy as np
# tensor于numpy转换
# Tensor与NumPy数组共享底层内存地址，修改一个会导致另一个的变化
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# NumPy Array 转化成 Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到 NumPy 的转换
# 使用.to 方法 可以将Tensor移动到任何设备中
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(a, device=device)
    x = a.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))