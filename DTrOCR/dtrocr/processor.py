from transformers import GPT2Tokenizer, AutoImageProcessor
from PIL import Image
from typing import List, Union
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput

class DTrOCRProcessor:
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = True, add_eos_token: bool = True):
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={"height": config.image_size[0], 'width': config.image_size[1]},
            use_fast=True
        )
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
        self.tokeniser.pad_token = self.tokeniser.bos_token
        self.tokeniser.add_eos_token = add_eos_token
        self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(self.tokeniser)

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = 'max_length',
        return_tensors: str = 'pt',  # Default to 'pt', but allow override
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        # Remove return_tensors from kwargs if provided to avoid duplication
        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k != 'return_tensors'}
        text_inputs = self.tokeniser(
            texts,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors,
            **tokenizer_kwargs
        ) if texts is not None else None

        image_inputs = self.vit_processor(
            images,
            input_data_format=input_data_format,
            return_tensors=return_tensors,
            **kwargs
        ) if images is not None else None

        target_lengths = None
        if text_inputs is not None and 'attention_mask' in text_inputs:
            target_lengths = text_inputs['attention_mask'].sum(dim=-1)

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            input_ids=text_inputs['input_ids'] if text_inputs is not None else None,
            attention_mask=text_inputs['attention_mask'] if text_inputs is not None else None,
            labels=text_inputs['input_ids'] if text_inputs is not None and return_labels else None,
            target_lengths=target_lengths if text_inputs is not None else None
        )

def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
    eos_token_ids = [self.eos_token_id] if self.add_eos_token else []
    output = bos_token_ids + token_ids_0 + eos_token_ids
    if token_ids_1 is None:
        return output
    return output + bos_token_ids + token_ids_1