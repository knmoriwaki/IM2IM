import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([8, 64, 128, 128], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(64, 1, kernel_size=[5, 5], padding=[2, 2], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()


