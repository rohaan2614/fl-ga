import torch

class GA:
    def __init__(self, 
                 seed: int, 
                 d: int, 
                 q: int, 
                 device: torch.device = torch.device("cpu")) -> None:
        self.seed = seed
        self.d = d
        self.q = q
        self.device = device
        torch.manual_seed(self.seed)
        
#         print(f'GA initialized with seed={seed}, d={d}, q={q}, device={device}')
        
    def G(self) -> torch.Tensor:
        torch.cuda.empty_cache()
        # Ensure the tensor is created directly on the specified device
        return torch.rand(self.d, self.q, device=self.device)
    
    def w(self, delta: torch.Tensor) -> torch.Tensor:
        torch.cuda.empty_cache()
        GT = self.G().t()  # Transpose the tensor
        return torch.matmul(GT, delta.to(self.device)) / self.q
    
    def delta(self, w: torch.Tensor) -> torch.Tensor:
        torch.cuda.empty_cache()
        return torch.matmul(self.G(), w.to(self.device))