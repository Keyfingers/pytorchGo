from __future__ import print_function
import torch
# PyTorch是什么?
# 基于Python的科学计算包，服务于以下两种场景:
#
# 作为NumPy的替代品，可以使用GPU的强大计算能力
# 提供最大的灵活性和高速的深度学习研究平台

# Tensors与Numpy中的 ndarrays类似，但是在PyTorch中 Tensors 可以使用GPU进行计算.

x = torch.empty(5, 3)
print(x)

# 创建一个随机初始化矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个0填充的矩阵，数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 创建tensor并使用现有的数据初始化
x = torch.tensor([5.5, 3])
print(x)

# 根据现有的张量创建张量
x = x.new_ones(5, 3, dtype=torch.double) # new_* 创建对象
print(x)
x = torch.randn_like(x, dtype=torch.float) # 覆盖dtype
print(x)                                   # 对象的size是相同的，只是值和类型不同
print(x.size())