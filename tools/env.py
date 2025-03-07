import torch
import mmcv

print(torch.__version__)  # 确保是 2.4.1+cu118
print(torch.version.cuda)  # 确保是 11.8
print(mmcv.__version__)  # 确保是 1.4.0
print(torch.cuda.is_available())  # True
