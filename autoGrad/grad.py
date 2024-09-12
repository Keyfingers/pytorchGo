import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# 在这个情形中，y不再是个标量。torch.autograd无法直接计算出完整的雅可比行列，
# 但是如果我们只想要vector-Jacobian product，只需将向量作为参数传入backward
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

# 如果.requires_grad=True但是你又不希望进行autograd的计算， 那么可以将变量包裹在 with torch.no_grad()中
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)