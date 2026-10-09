"""
one off script that builds model_gen_onnx/model_quantized.onnx, the generative video detector, about 200mb.
needs torch, transformers and onnx just for the export. the server itself only needs onnxruntime

    pip install torch transformers onnx onnxscript
    python export_generative_model.py
"""
import os
import torch
from transformers import AutoModelForImageClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

SOURCE = "haywoodsloan/ai-image-detector-deploy"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_gen_onnx")
FP32 = os.path.join(OUT_DIR, "model.onnx")
INT8 = os.path.join(OUT_DIR, "model_quantized.onnx")

os.makedirs(OUT_DIR, exist_ok=True)
model = AutoModelForImageClassification.from_pretrained(SOURCE).eval()
torch.onnx.export(
    model, (torch.randn(1, 3, 256, 256),), FP32,
    input_names=["pixel_values"], output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "b"}, "logits": {0: "b"}},
    opset_version=17, dynamo=False,
)
quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
os.remove(FP32)
print(f"wrote {INT8} ({os.path.getsize(INT8) / 1e6:.0f} MB)")
