"""
Placeholder for __init__.py
"""
# Run this to verify your environment is healthy
# python -c "from src.engine import verify_env; verify_env()"

def verify_env():
    import torch
    import cv2
    import ultralytics
    import sklearn
    import streamlit
    from loguru import logger

    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"Ultralytics: {ultralytics.__version__}")
    logger.success("Environment is ready for RapidAid!")