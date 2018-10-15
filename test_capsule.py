import torch as th

from capsule_layer import CapsuleLayer
from dgl_capsule_batch import DGLBatchCapsuleLayer

device = 'cuda'
th.manual_seed(12)
# th.cuda.seed(12)
x = th.randn((128, 8, 1152)).to(device)
W = th.randn(1152, 10, 16, 8).to(device)

model2 = CapsuleLayer(in_unit=8, in_channel=1152, num_unit=10, use_routing=True, unit_size=16, num_routing=3,
                      cuda_enabled=True).to("cuda")
model2.weight.data = W.clone().unsqueeze(0)
kkk2 = model2(x)
print(kkk2.norm())

model1 = DGLBatchCapsuleLayer(in_unit=8, in_channel=1152, num_unit=10, use_routing=True, unit_size=16, num_routing=3,
                              cuda_enabled=True).to("cuda")
model1.weight.data = W.clone()
kkk1 = model1(x)
print(kkk1.norm())

print((kkk1 - kkk2).norm())
# kkk.backward()
# print(model.W.grad.norm())
# print(kkk.shape)
