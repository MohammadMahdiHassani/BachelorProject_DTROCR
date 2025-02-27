import torch
from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class DTrOCRModelOutput:
    hidden_states: torch.FloatTensor
    past_key_values: torch.FloatTensor
    encoder_outputs: Optional[torch.FloatTensor] = None  # For RNNT encoder output
    prediction_outputs: Optional[torch.FloatTensor] = None  # For RNNT prediction network output

@dataclass
class DTrOCRLMHeadModelOutput:
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None
    past_key_values: Optional[torch.FloatTensor] = None
    rnnt_logits: Optional[torch.FloatTensor] = None  # For RNNT joint network output

@dataclass
class DTrOCRProcessorOutput:
    pixel_values: Optional[torch.FloatTensor] = None
    input_ids: Optional[Union[torch.LongTensor, List[int]]] = None
    attention_mask: Optional[Union[torch.FloatTensor, List[int]]] = None
    labels: Optional[Union[torch.LongTensor, List[int]]] = None
    target_lengths: Optional[torch.LongTensor] = None  # For RNNT loss