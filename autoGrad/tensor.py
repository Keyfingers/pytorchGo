import torch

# 结果y已经被计算出来了，所以，grad_fn已经被自动生成了。
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# .requires_grad_( ... ) 可以改变现有张量的 requires_grad属性。 如果没有指定的话，默认输入的flag是 False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)