import torch

# 创建一个2x2的张量，并设置requires_grad=True来追踪它的计算历史
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y)
# 打印y的梯度函数，它记录了如何计算y的梯度
print(y.grad_fn)

z = y * y * 3
# 计算z的均值，创建一个新的标量张量out
out = z.mean()
print(z, out)

# 如果requires_grad没有指定的话，默认输入的flag是 False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
# .requires_grad_( ... ) 可以改变现有张量的 requires_grad属性。
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))
out.backward()
print(x.grad)