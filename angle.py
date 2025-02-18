import os
import numpy as np
import cv2
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
from time import time
from PIL import Image
from torchvision.transforms import transforms

# Visualization imports
from vit_utils.visualization import draw_points_and_skeleton, joints_dict
from vit_utils.top_down_eval import keypoints_from_heatmaps


# 1️⃣  Load and Convert ONNX to TensorRT Engine
def export_engine(onnx_path, im, engine_path, half=False, dynamic=True, workspace=4, verbose=False, prefix='Tensorrt'):
    """Converts an ONNX model to a TensorRT engine (compatible with TensorRT 10+)."""
    
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # 🔹 FIX: Use set_memory_pool_limit for TensorRT 10+
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    if not parser.parse_from_file(str(onnx_path)):
        print(f"Failed to parse ONNX file: {onnx_path}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))  # Print parsing errors
        raise RuntimeError(f"ONNX parsing failed: {onnx_path}")

    # Display model inputs/outputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    for inp in inputs:
        print(f"{prefix} input: {inp.name}, Shape: {inp.shape}, Type: {inp.dtype}")
    for out in outputs:
        print(f"{prefix} output: {out.name}, Shape: {out.shape}, Type: {out.dtype}")

    # Handle dynamic batch size
    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    # Enable FP16 if supported
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    # 🔹 FIX: Use build_serialized_network() for TensorRT 10+
    print(f"{prefix} building FP{16 if half else 32} engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized TensorRT engine!")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✅ TensorRT engine saved at {engine_path}")
    return True





# 2️⃣  Load TensorRT Engine
def load_engine(trt_runtime, engine_path):
    """Loads the TensorRT engine."""
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)


# 3️⃣  Allocate Buffers
def allocate_buffers(engine):
    """Allocates memory for inputs/outputs with dynamic shapes."""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        # Get shape and dtype dynamically
        shape = engine.get_tensor_shape(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Calculate size of the buffer
        size = trt.volume(shape) * engine.get_max_batch_size()
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    return inputs, outputs, bindings, stream



# 4️⃣  Run Inference
def do_inference(context, bindings, inputs, outputs, stream):
    """Runs inference on the TensorRT engine."""
    for inp in inputs:
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()
    return [out["host"] for out in outputs]


# 5️⃣  Convert ONNX to TensorRT
ONNX_PATH = "E:/vitpose-l-ap10k.onnx"
TRT_PATH = ONNX_PATH.replace('.onnx', '.engine')

C, H, W = 3, 256, 192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_input = torch.randn(1, C, H, W).to(device)

export_engine(ONNX_PATH, sample_input, TRT_PATH, half=False, dynamic=True)


# 6️⃣  Load TensorRT Model for Inference
logger = trt.Logger(trt.Logger.ERROR)
trt_runtime = trt.Runtime(logger)

print(f"Loading TensorRT engine from {TRT_PATH}...")
trt_engine = load_engine(trt_runtime, TRT_PATH)

# Allocate buffers
inputs, outputs, bindings, stream = allocate_buffers(trt_engine)

# Create execution context
context = trt_engine.create_execution_context()


# 7️⃣  Prepare Image for Inference
IMG_PATH = "E:\lame.webp"
img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
org_h, org_w = img.shape[:2]

# Preprocess Image
img_input = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
img_input = img_input.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0

# Copy to GPU memory
np.copyto(inputs[0]["host"], img_input.ravel())


# 8️⃣  Run Inference
tic = time()
heatmaps = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0]
heatmaps = heatmaps.reshape((1, 25, 64, 48))
elapsed_time = time() - tic

print(f"Output size: {heatmaps.shape}, Time elapsed: {elapsed_time:.4f}s, FPS: {1/elapsed_time:.1f}")


# 9️⃣  Postprocess & Visualization
points, prob = keypoints_from_heatmaps(
    heatmaps=heatmaps, 
    center=np.array([[org_w // 2, org_h // 2]]), 
    scale=np.array([[org_w, org_h]]), 
    unbiased=True, use_udp=True
)

points = np.concatenate([points[:, :, ::-1], prob], axis=2)

# Draw Keypoints
for pid, point in enumerate(points):
    img_vis = draw_points_and_skeleton(
        img.copy(), point, joints_dict()["coco"]["skeleton"], person_index=pid,
        points_color_palette="gist_rainbow", skeleton_color_palette="jet",
        points_palette_samples=10, confidence_threshold=0.4
    )
    
    plt.figure(figsize=(5, 10))
    plt.imshow(img_vis)
    plt.title("Result")
    plt.axis("off")
    plt.show() dont convert the file we have file already just run the inference
