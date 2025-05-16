from dataclasses import dataclass, field
from .base_backbone_config import BaseBackboneConfig
from typing import Union, List, Optional
from torch import nn


@dataclass
class MLPBackboneConfig(BaseBackboneConfig):
    input_dim: Optional[int] = field(default=None)
    # //////////////////////
    hidden_dims: Union[int, List[int]] = field(default=128)
    # 
    output_dim: Optional[int] = field(default=None)
    n_hiddens: int = field(default=2)
    activation: str = field(default='ReLU')
    use_batch_norm: bool = field(default=True)

    def __post_init__(self):
        self.name = "mlp"

        if isinstance(self.hidden_dims, int):
            if self.n_hiddens > 1:
                self.hidden_dims = [
                    self.hidden_dims for _ in range(self.n_hiddens - 1)]
            else:
                self.output_dim = self.hidden_dims
                self.hidden_dims = []

        if self.input_dim is None:
            raise TypeError(
                "__init__ missing 1 required positional argument: 'input_dim'")

        if self.output_dim is None:
            self.output_dim = self.hidden_dims[-1]

        if not hasattr(nn, self.activation):
            raise ValueError(
                f"{self.activation} is not a valid activation of torch.nn")



# from dataclasses import dataclass, field
# from .base_backbone_config import BaseBackboneConfig
# from typing import List, Optional, Any
# from torch import nn


# @dataclass
# class MLPBackboneConfig(BaseBackboneConfig):
#     input_dim: Optional[int] = field(default=None)
    
#     # Use Any here to avoid Union error in config validation.
#     hidden_dims: Any = field(default=128)  # Will normalize to List[int] in __post_init__
    
#     output_dim: Optional[int] = field(default=None)
#     n_hiddens: int = field(default=2)
#     activation: str = field(default='ReLU')
#     use_batch_norm: bool = field(default=True)

#     def __post_init__(self):
#         self.name = "mlp"

#         # Normalize hidden_dims to a list of ints
#         if isinstance(self.hidden_dims, int):
#             if self.n_hiddens > 1:
#                 self.hidden_dims = [self.hidden_dims] * (self.n_hiddens - 1)
#             else:
#                 self.output_dim = self.hidden_dims
#                 self.hidden_dims = []
#         elif isinstance(self.hidden_dims, list):
#             # Optional: Validate all elements are ints
#             if not all(isinstance(x, int) for x in self.hidden_dims):
#                 raise ValueError("All elements in hidden_dims must be integers.")
#         else:
#             raise TypeError("hidden_dims must be either int or list of ints")

#         if self.input_dim is None:
#             raise TypeError(
#                 "__init__ missing 1 required positional argument: 'input_dim'")

#         # If output_dim is not provided, set it to the last element of hidden_dims or keep as is
#         if self.output_dim is None:
#             if self.hidden_dims:
#                 self.output_dim = self.hidden_dims[-1]
#             else:
#                 # If hidden_dims is empty, fallback to input_dim (or raise)
#                 self.output_dim = self.input_dim

#         # Validate activation exists in torch.nn
#         if not hasattr(nn, self.activation):
#             raise ValueError(
#                 f"{self.activation} is not a valid activation of torch.nn")
