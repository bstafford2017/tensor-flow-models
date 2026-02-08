def normalize_data(data, scaler_path):
    """Normalizes the data using Min-Max scaling."""
    min_val = data.min()
    max_val = data.max()
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

def denormalize_data(normalized_data, scaler_path):
    """Denormalizes the data back to its original scale."""
    min_val = original_data.min()
    max_val = original_data.max()
    denormalized = normalized_data * (max_val - min_val) + min_val
    return denormalized