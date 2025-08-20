import torch

def get_entropy_of_dataset(matrix: torch.Tensor):
    """
    Calculate the entropy of the entire dataset (for PyTorch)
    Last column in 'matrix' is assumed to be the target variable.
    """
    if matrix.shape[0] == 0:
        return 0.0
    label_column = matrix[:, -1]
    classes, occurrences = torch.unique(label_column, return_counts=True)
    total_rows = matrix.shape[0]
    probs = occurrences.float() / total_rows
    entropy_value = 0.0
    for p in probs:
        if p > 0:
            entropy_value -= p * torch.log2(p)
    return float(entropy_value)

def get_avg_info_of_attribute(matrix: torch.Tensor, feature: int):
    """
    Calculate the average information (weighted entropy) of an attribute (PyTorch).
    """
    if matrix.shape[0] == 0 or feature < 0 or feature >= matrix.shape[1] - 1:
        return 0.0
    feature_col = matrix[:, feature]
    total_rows = matrix.shape[0]
    distinct_vals = torch.unique(feature_col)
    avg_information = 0.0
    for val in distinct_vals:
        mask = (feature_col == val)
        subset_matrix = matrix[mask]
        ratio = subset_matrix.shape[0] / total_rows
        if subset_matrix.shape[0] > 0:
            sub_entropy = get_entropy_of_dataset(subset_matrix)
            avg_information += ratio * sub_entropy
    return float(avg_information)

def get_information_gain(matrix: torch.Tensor, feature: int):
    """
    Calculate information gain for an attribute in the dataset (PyTorch).
    """
    if matrix.shape[0] == 0:
        return 0.0
    dataset_entropy = get_entropy_of_dataset(matrix)
    avg_information = get_avg_info_of_attribute(matrix, feature)
    gain_value = dataset_entropy - avg_information
    return round(float(gain_value), 4)

def get_selected_attribute(matrix: torch.Tensor):
    """
    Return:
      - dict: {attribute_index: information_gain}
      - int: index of attribute with highest info gain
    for PyTorch tensors.
    """
    if matrix.shape[0] == 0 or matrix.shape[1] <= 1:
        return {}, -1
    total_features = matrix.shape[1] - 1
    info_gains = {}
    for i in range(total_features):
        info_gains[i] = get_information_gain(matrix, i)
    if not info_gains:
        return {}, -1
    best_feature = max(info_gains, key=info_gains.get)
    return info_gains, best_feature
