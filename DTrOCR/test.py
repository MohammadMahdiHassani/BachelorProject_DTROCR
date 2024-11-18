from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor


from PIL import Image

config = DTrOCRConfig()
model = DTrOCRLMHeadModel(config)
processor = DTrOCRProcessor(DTrOCRConfig())

model.eval()        # set model to evaluation mode for deterministic behaviour
path_to_image = "hello.png"  # path to image file

inputs = processor(
    images=Image.open(path_to_image).convert('RGB'),
    texts=processor.tokeniser.bos_token,
    return_tensors="pt"
)

model_output = model.generate(
    inputs=inputs, 
    processor=processor, 
    num_beams=3,    # defaults to 1 if not specified
    use_cache=True  # defaults to True if not specified
)

predicted_text = processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
print("text:\n")
print(predicted_text)