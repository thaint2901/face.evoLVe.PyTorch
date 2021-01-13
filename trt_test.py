import cv2
import numpy as np
import torch
import timeit
from pretrained.model_irse import IR_50
from arcface._C import Engine

input_size = [112, 112]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = IR_50(input_size)
model.load_state_dict(torch.load("pretrained/backbone_ir50_asia.pth"))
model.eval()
model = model.to(device)

model_trt = Engine.load("arcfaceR50_asia.plan")
batch_size = 2

img = cv2.imread("cpp/BruceLee.jpg")
img = cv2.resize(img, (112, 112))
img = img.transpose(2, 0, 1)
img = img.astype(np.float32)
img = (img - 127.5) / 128.0
img = np.stack([img] * batch_size)

input_tensor = torch.from_numpy(img.copy()).to("cuda")
input_tensor_trt = torch.from_numpy(img.copy()).to("cuda")
# for _ in range(1000):
t0 = timeit.default_timer()

print(input_tensor.shape)
embs = model(input_tensor)
embs = embs.detach().cpu().numpy()

embs_trt = model_trt(input_tensor_trt)
embs_trt = embs_trt[0].cpu().numpy()
t1 = timeit.default_timer()
for i, j in zip(embs_trt[0], embs[0]):
    print(i,j)
print(np.sum(np.abs(embs-embs_trt)))
# print(embs, embs.shape)