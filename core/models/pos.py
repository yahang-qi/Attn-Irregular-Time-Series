import torch

def positional_encoding(time_steps, d_embed, device='cuda'):
    """
    Compute positional encodings for a batch of time steps.
    
    Args:
        time_steps (torch.Tensor): Tensor of shape (bsz, max_len) containing time step values.
        d_embed (int): Dimensionality of the embeddings.
        device (str): Device to perform the computation on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Positional encodings of shape (bsz, max_len, d_embed).
    """
    
    # Create positional indices for each time step
    pos = time_steps.unsqueeze(2).to(device)  # Shape: (bsz, max_len, 1)
    i = torch.arange(d_embed, device=device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, d_embed)
    
    # Compute angles for positional encoding
    angles = pos / (10000 ** (2 * torch.div(i, 2, rounding_mode='trunc') / d_embed))  # Shape: (bsz, max_len, d_embed)
    
    # Initialize positional encoding tensor
    pos_encoding = torch.zeros_like(angles).to(device)
    
    # Apply sin to even indices and cos to odd indices
    pos_encoding[:, :, 0::2] = torch.sin(angles[:, :, 0::2])  # Even indices
    pos_encoding[:, :, 1::2] = torch.cos(angles[:, :, 1::2])  # Odd indices
    
    return pos_encoding
