"""Device selection utility."""
from typing import Optional
import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda:0")  #在jyz服务器测试，改为cuda:0
    return torch.device("cpu")
