import onnxruntime as rt
sess = rt.InferenceSession("arcfaceR50_asia.onnx")
print("====INPUT====")
for i in sess.get_inputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))
print("====OUTPUT====")
for i in sess.get_outputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))