import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from tqdm import tqdm
from src.components.model_loader.clip_loader import CLIPLoader

# --- Dataset class ---
class ImageCaptionDataset(Dataset):
    def __init__(self, data_list, processor, image_size=(224, 224)):
        """
        data_list: list of tuples (image_path, caption)
        processor: CLIPProcessor from loader
        """
        self.data = data_list
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        return {"image": image, "caption": caption}

# --- Trainer class ---
class CLIPFineTuner:
    def __init__(self, clip_loader: CLIPLoader, lr=5e-6, batch_size=16, epochs=3, device=None):
        self.clip_loader = clip_loader
        self.model = clip_loader.model
        self.processor = clip_loader.processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        # Freeze most layers if you want partial fine-tuning
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # Only train the projection layers
        # for param in self.model.visual_projection.parameters():
        #     param.requires_grad = True
        # for param in self.model.text_projection.parameters():
        #     param.requires_grad = True

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CosineEmbeddingLoss()  # CLIP uses cosine similarity

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            pbar = tqdm(dataloader)
            for batch in pbar:
                images, captions = batch["image"], batch["caption"]
                images = list(images)
                text_inputs = self.processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
                image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                # Forward pass
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)

                # Normalize embeddings
                image_features = nn.functional.normalize(image_features, p=2, dim=-1)
                text_features = nn.functional.normalize(text_features, p=2, dim=-1)

                # CLIP loss: cosine similarity
                # target=1 means matching pairs
                targets = torch.ones(image_features.size(0)).to(self.device)
                loss = self.loss_fn(image_features, text_features, targets)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f"Loss: {loss.item():.4f}")

        print("[INFO] Fine-tuning complete!")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved at {path}")