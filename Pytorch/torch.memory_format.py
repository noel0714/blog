import torch

N, C, H, W = 10, 3, 32, 32

# contiguous memory format
x = torch.empty(N, C, H, W)
print(f"x.shape = {x.shape}, x.stride() = {x.stride()}")

# channels_last memory format
x = x.contiguous(memory_format=torch.channels_last)
print(f"x.shape = {x.shape}, x.stride() = {x.stride()}")

# alternative option
# x = x.to(memory_format=torch.channels_last)
# print(f"x.shape = {x.shape}, x.stride() = {x.stride()}")

# clone() preserves memory format
y = x.clone()
print(f"y.shape = {y.shape}, y.stride() = {y.stride()}")

# pointwise operators preserves memory format
z = x + y
print(f"z.shape = {z.shape}, z.stride() = {z.stride()}")

'''
results

x.shape = torch.Size([10, 3, 32, 32]), x.stride() = (3072, 1024, 32, 1)
x.shape = torch.Size([10, 3, 32, 32]), x.stride() = (3072, 1, 96, 3)
y.shape = torch.Size([10, 3, 32, 32]), y.stride() = (3072, 1024, 32, 1)
z.shape = torch.Size([10, 3, 32, 32]), z.stride() = (3072, 1, 96, 3)
'''
