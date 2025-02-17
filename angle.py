import os
import tensorrt as trt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_engine(onnx_path, engine_path, precision="fp16", min_batch=1, opt_batch=1, max_batch=1):
    """
    Convert ONNX model to TensorRT engine with dynamic batch size support
    """
    logger.info(f"Converting {onnx_path} to TensorRT engine...")
    logger.info(f"TensorRT version: {trt.__version__}")
    
    # Initialize TensorRT engine builder
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network definition
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(f"Parser error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX")
    
    # Print network info
    logger.info(f"Network inputs: {network.num_inputs}")
    logger.info(f"Network outputs: {network.num_outputs}")
    
    # Create optimization profile
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)  # 8GB
    
    profile = builder.create_optimization_profile()
    
    # Set input dimensions for dynamic batching
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    min_shape = (min_batch, input_shape[1], input_shape[2], input_shape[3])
    opt_shape = (opt_batch, input_shape[1], input_shape[2], input_shape[3])
    max_shape = (max_batch, input_shape[1], input_shape[2], input_shape[3])
    
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")
    
    # Enable all available tactics
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    
    # Build engine
    logger.info("Building engine... (this may take several minutes)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build engine - returned None")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        logger.info(f"Successfully saved engine to {engine_path}")
        
        # Try to create runtime engine to verify
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            raise RuntimeError("Failed to deserialize engine")
        logger.info("Engine successfully verified")
        
        return serialized_engine
        
    except Exception as e:
        logger.error(f"Engine build failed: {str(e)}")
        raise

if __name__ == "__main__":
    onnx_model = "E:/vitpose/vitpose-l-ap10k.onnx"
    engine_path = "vitpose.engine"
    
    try:
        build_engine(
            onnx_path=onnx_model,
            engine_path=engine_path,
            precision="fp16",
            min_batch=1,
            opt_batch=1,
            max_batch=1
        )
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise
