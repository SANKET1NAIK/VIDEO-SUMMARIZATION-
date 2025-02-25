import torch
import tensorrt as trt
import os

# Model paths
ONNX_PATH = "C:/Users/hp/Downloads/vitpose-l-ap10k.onnx"
TRT_PATH = ONNX_PATH.replace('.onnx', '.engine')

# Model configuration
C, H, W = (3, 256, 192)
input_names = ["input_0"]
output_names = ["output_0"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy input tensor
inputs = torch.randn(1, C, H, W).to(device)

# Dynamic axes configuration
dynamic_axes = {
    'input_0': {0: 'batch_size'},
    'output_0': {0: 'batch_size'}
}

def export_engine(onnx, im, file, workspace=4, verbose=False, prefix='Tensorrt'):
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    
    # Initialize builder and config
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    
    # Create network
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    
    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    # Process inputs and outputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    # Print input and output details
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    # Set profile for dynamic shapes
    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(
            inp.name, 
            (1, *im.shape[1:]),                           # min shape
            (max(1, im.shape[0] // 2), *im.shape[1:]),    # optimal shape
            im.shape                                       # max shape
        )
    config.add_optimization_profile(profile)

    # Force FP32 precision - no FP16 configuration
    print(f'{prefix} building FP32 engine')
    
    # Build and save engine
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        plan = builder.build_engine(network, config)
        serialized_engine = plan.serialize()
        plan.destroy()
    
    if serialized_engine is None:
        raise RuntimeError(f'{prefix} failed to build TensorRT engine')
    
    with open(file, 'wb') as f:
        f.write(serialized_engine)
    
    return True

# Check if ONNX file exists
if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"ONNX file not found at {ONNX_PATH}")

# Convert to TensorRT
print(f"Converting {ONNX_PATH} to TensorRT engine (FP32)...")
success = export_engine(
    onnx=ONNX_PATH,
    im=inputs,
    file=TRT_PATH,
    verbose=False
)

if success:
    print(f"Successfully converted to TensorRT engine (FP32): {TRT_PATH}")
else:
    print("Conversion failed!")
