import numpy as np
import io
import torch
from pretrained.model_irse import IR_50

input_size = [112, 112]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_onnx = "arcfaceR50_asia.onnx"

model = IR_50(input_size)
model.load_state_dict(torch.load("pretrained/backbone_ir50_asia.pth"))
model.eval()
model = model.to(device)

print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ['input_1']
output_names = ['output_1']

onnx_bytes = io.BytesIO()
zero_input = torch.zeros([1, 3, 112, 112])
zero_input = zero_input.to(device)
dynamic_axes = {input_names[0]: {0:'batch'}}
for _, name in enumerate(output_names):
    dynamic_axes[name] = dynamic_axes[input_names[0]]
extra_args = {'opset_version': 10, 'verbose': False,
                'input_names': input_names, 'output_names': output_names,
                'dynamic_axes': dynamic_axes}
torch.onnx.export(model, zero_input, onnx_bytes, **extra_args)
with open(output_onnx, 'wb') as out:
    out.write(onnx_bytes.getvalue())