from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from dtrocr.config import DTrOCRConfig
from dtrocr.processor import DTrOCRProcessor
from dtrocr.model import DTrOCRLMHeadModel
import tqdm

# Dataset
class IAMWordsDataset(Dataset):
    def __init__(self, split_file, processor, root_dir='iam_words'):
        self.processor = processor
        self.data = self._load_data(f'{root_dir}/splits/{split_file}')

    def _load_data(self, split_file):
        with open(split_file, 'r') as f:
            ids = {line.strip() for line in f}
        # Replace with your Word list loading logic
        words = [
            {'id': 'a01-000u', 'file_path': 'iam_words/words/a01/a01-000u/a01-000u-00-00.png', 'transcription': 'A'},
            {'id': 'a01-000u', 'file_path': 'iam_words/words/a01/a01-000u/a01-000u-00-01.png', 'transcription': 'MOVE'},
        ]
        return [(w['file_path'], w['transcription']) for w in words if w['id'] in ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # Qwen2.5-VL expects RGB
        processed = self.processor(images=image, texts=text, return_labels=True)
        return processed.__dict__

# Setup
config = DTrOCRConfig(
    image_size=(448, 448),
    num_channels=3,
    hidden_size=1280  # Optional: match Qwen2.5-VL without projection
)
processor = DTrOCRProcessor(config)
model = DTrOCRLMHeadModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = IAMWordsDataset('train.uttlist', processor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Smaller batch size due to model size

# Training loop
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(3):
    model.train()
    for inputs in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = model(**inputs)
        optimiser.zero_grad()
        outputs.loss.backward()
        optimiser.step()
    print(f"Epoch {epoch + 1}, Loss: {outputs.loss.item()}")