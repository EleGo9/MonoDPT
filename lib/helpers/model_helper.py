from lib.models.monodpt import build_monodpt

# Model registry - maps model_type from config to the appropriate builder function
MODEL_REGISTRY = {
    'MonoDPT': build_monodpt,
    'monodpt': build_monodpt,  # Case-insensitive support
    # Add more model types here as they are implemented:
    # 'SomeOtherModel': build_some_other_model,
}


def build_model(cfg):
    """
    Build model based on cfg.model_type from the config file.

    Args:
        cfg: Configuration object containing model_type field (defaults to "MonoDPT")

    Returns:
        Built model and criterion

    Raises:
        ValueError: If model_type is not registered
    """
    model_type = cfg.model_type

    if model_type not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available models: {available_models}"
        )

    builder_fn = MODEL_REGISTRY[model_type]
    return builder_fn(cfg)
