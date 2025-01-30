import torch
import os

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

# print(os.environ.keys())

#print(torch.cuda.is_available())
#x = torch.cuda.current_device()
#print(torch.cuda.get_device_name(x))
