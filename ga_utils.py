import torch

def flatten_vector(state_dict, device):
    flat_vector = []
    shapes = {}

    for name, tensor in state_dict.items():
        flattened_tensor = tensor.view(-1)  # Flatten the tensor into a 1D vector
        flat_vector.append(flattened_tensor)
        shapes[name] = tensor.shape  # Store the original shape

    # Concatenate all flattened tensors into a single 1D vector
    flat_vector = torch.cat(flat_vector)

    return flat_vector.to(device), shapes

def vector_to_state_dict(flat_vector, shapes, torchvision_model=False):
    reconstructed_state_dict = {}
    current_index = 0

    for name, shape in shapes.items():
        # Calculate the number of elements in the tensor
        if torchvision_model:
            num_elements = int(torch.prod(torch.tensor(shape)).item())
        else:
            num_elements = torch.prod(torch.tensor(shape))
        # Extract the corresponding portion from the flat vector
        flattened_tensor = flat_vector[current_index:current_index + num_elements]
        # Reshape the 1D tensor back to its original shape
        tensor = flattened_tensor.view(shape)
        # Add the tensor back to the state_dict
        reconstructed_state_dict[name] = tensor
        # Update the current index
        current_index += num_elements

    return reconstructed_state_dict

def normalize(tensor: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    # Calculate the mean and standard deviation
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    
    # Apply Z-Score normalization
    normalized_tensor = (tensor - tensor_mean) / tensor_std
    
    # Min and max of the normalized tensor (Z-Score normalized tensors are usually centered around 0)
    tensor_min = normalized_tensor.min()
    tensor_max = normalized_tensor.max()
    
    # Scale to the desired range [min_val, max_val]
    scaled_tensor = (normalized_tensor - tensor_min) / (tensor_max - tensor_min)  # Scale to [0, 1]
    scaled_tensor = scaled_tensor * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
    
    return scaled_tensor

def scale_to_match_range(delta: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    min_delta, max_delta = delta.min(), delta.max()
    min_gw, max_gw = gw.min(), gw.max()
    
    norm_gw = (gw - min_gw) / (max_gw - min_gw)
    
    # Scale norm_gw to the range of delta
    scaled_gw = norm_gw * (max_delta - min_delta) + min_delta
    
    return scaled_gw

def center_around_delta(delta: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    # Calculate the mean of tensor1
    mean_tensor1 = delta.mean()
    
    # Calculate the mean of tensor2
    mean_tensor2 = gw.mean()
    
    # Compute the shift needed to center tensor2 around the mean of tensor1
    shift = mean_tensor1 - mean_tensor2
    
    # Apply the shift to tensor2
    shifted_tensor = gw + shift
    
    return shifted_tensor

def match_spread(delta: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    # Calculate the mean and standard deviation of both tensors
    mean_delta = delta.mean()
    std_delta = delta.std()
    
    mean_gw = gw.mean()
    std_gw = gw.std()
    
    # Check if the means of delta and gw are the same
    if not torch.isclose(mean_delta, mean_gw):
        raise ValueError("The means of delta and gw must be the same.")
    
    # Scale gw to match the standard deviation of delta
    scaled_gw = (gw - mean_gw) / std_gw  # Normalize gw to mean 0 and std 1
    adjusted_gw = scaled_gw * std_delta + mean_delta  # Scale to match the std of delta
    
    return adjusted_gw