import torch
import numpy as np
import onnxruntime as ort

JIT_PATH = "models/best_model_new.pt"
ONNX_PATH = "models/best_model_new.onnx"
INFER_HEIGHT = 256
INFER_WIDTH = 256


def convert():
    device = torch.device("cpu")

    print(f"Loading JIT model from {JIT_PATH}")
    model = torch.jit.load(JIT_PATH, map_location=device)
    model.eval()

    dummy_input = torch.randn(1, 3, INFER_HEIGHT, INFER_WIDTH, device=device)

    print(f"Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    print("Export done.")

    print("Verifying ONNX model...")
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    x_np = dummy_input.numpy()
    result = session.run(["output"], {"input": x_np})[0]
    print(f"ONNX output: shape={result.shape}, dtype={result.dtype}")

    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    max_diff = np.abs(result - pt_out).max()
    print(f"Max diff PyTorch vs ONNX: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("Conversion successful, diff is within tolerance.")
    else:
        print(f"WARNING: diff is large ({max_diff:.6f}), check the model.")


if __name__ == "__main__":
    convert()
