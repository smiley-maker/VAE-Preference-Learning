from src.VAE.utils.imports import *

class VAE_Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)