import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Generator Model Definition (Same as Training)
class Generator(torch.nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = torch.nn.Embedding(num_classes, num_classes)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + num_classes, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 28*28),
            torch.nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embed = self.label_embed(labels)
        x = torch.cat([noise, label_embed], dim=-1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Load Trained Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator()
    model.load_state_dict(torch.load('generator_final.pth', map_location=device))
    model.eval()
    return model

# Streamlit UI
st.title("Handwritten Digit Generator")
digit = st.selectbox("Select a digit:", options=list(range(10)), index=5)
n_images = 5

if st.button("Generate Images"):
    model = load_model()
    noise = torch.randn(n_images, 100)
    labels = torch.tensor([digit] * n_images)
    
    with torch.no_grad():
        generated = model(noise, labels).detach().cpu()
    
    # Display images
    cols = st.columns(n_images)
    for i in range(n_images):
        img = generated[i].squeeze().numpy()
        img = (img * 0.5 + 0.5) * 255  # Denormalize
        img = Image.fromarray(img.astype('uint8'))
        cols[i].image(img, caption=f"Digit {digit}", width=100)