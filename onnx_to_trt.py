import torch
import tensorrt as trt
import os

# Model paths
ONNX_PATH = "C:/Users/hp/Downloads/vitpose-l-ap10k.onnx"
TRT_PATH = ONNX_PATH.replace('.onnx', '.engine')

# Model configuration
C, H, W = (3, 256, 192)  # Channels, Height, Width
input_names = ["input_0"]
output_names = ["output_0"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy input tensor
inputs = torch.randn(1, C, H, W).to(device)

# Dynamic axes configuration (not directly used here but kept for reference)
dynamic_axes = {
    'input_0': {0: 'batch_size'},
    'output_0': {0: 'batch_size'}
}

def export_engine(onnx, im, file, workspace=4, verbose=False, prefix='Tensorrt'):
    """
    Convert an ONNX model to a TensorRT engine.
    
    Args:
        onnx (str): Path to the ONNX model file
        im (torch.Tensor): Dummy input tensor for shape inference
        file (str): Path to save the TensorRT engine
        workspace (int): Workspace size in GB (default: 4GB)
        verbose (bool): Enable verbose logging
        prefix (str): Prefix for logging messages
    Returns:
        bool: True if conversion succeeds, raises exception otherwise
    """
    # Set up TensorRT logger
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    
    # Initialize builder and config
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # Convert GB to bytes
    
    # Create network with explicit batch dimension
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'Failed to load ONNX file: {onnx}')

    # Print input and output details
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'{prefix} input "{inp.name}" with shape {inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'{prefix} output "{out.name}" with shape {out.shape} {out.dtype}')

    # Set dynamic shapes using optimization profile
    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(
            inp.name,
            (1, *im.shape[1:]),                           # Min shape (batch=1)
            (max(1, im.shape[0] // 2), *im.shape[1:]),    # Optimal shape
            im.shape                                       # Max shape
        )
    config.add_optimization_profile(profile)

    # Build FP32 engine
    print(f'{prefix} building FP32 engine')
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f'{prefix} failed to build TensorRT engine')
    
    # Save the engine to file
    with open(file, 'wb') as f:
        f.write(serialized_engine)
    
    return True

# Main execution
if __name__ == "__main__":
    # Check if ONNX file exists
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX file not found at {ONNX_PATH}")

    # Convert to TensorRT
    print(f"Converting {ONNX_PATH} to TensorRT engine (FP32)...")
    try:
        success = export_engine(
            onnx=ONNX_PATH,
            im=inputs,
            file=TRT_PATH,
            verbose=False  # Set to True for detailed logs
        )
        if success:
            print(f"Successfully converted to TensorRT engine (FP32): {TRT_PATH}")
    except Exception as e:
        print(f"Conversion failed! Error: {str(e)}")
