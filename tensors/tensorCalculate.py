import torch

x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
y = torch.rand(5, 3)
# 1、直接相加
print(x + y)
# 2、torch.add(a,b)相加
print(torch.add(x, y))

# 提供输出tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.
y.add_(x)
print(y)

# 使用与NumPy索引方式相同的操作来进行对张量的操作
print(x[:,1])

# torch.view 与Numpy的reshape类似，torch.view 函数用于重塑张量的形状，而不改变其数据
x = torch.randn(4, 4)
# 使用view将张量x重塑为一个包含16个元素的一维张量
y = x.view(16)
# 使用view将张量x重塑为一个2x8的二维张量，-1 表示这个维度的大小会自动计算，以便保持元素总数不变 2=（16/8）
z = x.view(-1,8)
print(x.size(), y.size(), z.size())

# 创建一个包含一个正态分布随机值的张量
x = torch.randn(1)
print(x)
# 将张量x中的值作为Python数值类型打印，.item() 会返回一个Python浮点数，例如 0.2345。如果张量包含多个元素，.item() 方法会引发错误，因为它只能用于单个元素张量。
print(x.item())