import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim
from PIL import Image, ImageFilter
import numpy as np
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import time

st.set_page_config(page_title="Digit Recognizer", layout="wide")

# Define improved CNN
class ImprovedMNISTCNN(nn.Module):
    def __init__(self):
        super(ImprovedMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc_dropout(x)
        x = self.fc3(x)
        return x

@st.cache_resource
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = ImprovedMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    epochs = 10
    best_acc = 0
    progress = st.progress(0)
    status = st.empty()
    acc_text = st.empty()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_mnist.pth')
        acc_text.text(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {acc:.2f}%")
        progress.progress((epoch + 1) / epochs)

    model.load_state_dict(torch.load('best_mnist.pth', map_location=device))
    status.text(f"Training complete! Best accuracy: {best_acc:.2f}%")
    return model, device

@st.cache_resource
def get_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ImprovedMNISTCNN().to(device)
        model.load_state_dict(torch.load('best_mnist.pth', map_location=device))
        model.eval()
        st.success("Loaded pre-trained model")
        return model, device
    except:
        st.warning("Training new model...")
        return train_model()

def preprocess_canvas(canvas_result):
    """Canvas already has white digit on black background. No inversion needed."""
    if canvas_result.image_data is None:
        return None, None
    img = canvas_result.image_data.astype(np.uint8)
    # Take alpha if RGBA, else convert
    if img.shape[-1] == 4:
        # Use alpha channel or convert to grayscale
        if np.all(img[:,:,:3] == img[:,:,0:1]):  # grayscale
            img = Image.fromarray(img[:,:,0]).convert('L')
        else:
            img = Image.fromarray(img).convert('L')
    else:
        img = Image.fromarray(img).convert('L')
    # Canvas already white on black, so no inversion
    # But we need to ensure digit is white (255) on black (0)
    # The canvas gives white on black, so fine.
    # Center and resize
    img_array = np.array(img)
    non_zero = np.where(img_array > 50)
    if len(non_zero[0]) == 0:
        return None, None  # blank
    y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
    x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
    # Add padding
    pad = 5
    y_min = max(0, y_min - pad)
    y_max = min(img_array.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(img_array.shape[1], x_max + pad)
    img = img.crop((x_min, y_min, x_max, y_max))
    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new('L', (28, 28), color=0)
    new_img.paste(img, (4, 4))
    new_img = new_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(new_img).unsqueeze(0)
    return tensor, new_img

def preprocess_upload(image, invert=True):
    """For uploaded images, assume black digit on white background, invert if needed."""
    if image.mode != 'L':
        image = image.convert('L')
    if invert:
        image = Image.eval(image, lambda x: 255 - x)  # make white on black
    # Same centering as above
    img_array = np.array(image)
    non_zero = np.where(img_array > 50)
    if len(non_zero[0]) == 0:
        return None, None
    y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
    x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
    pad = 5
    y_min = max(0, y_min - pad)
    y_max = min(img_array.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(img_array.shape[1], x_max + pad)
    image = image.crop((x_min, y_min, x_max, y_max))
    image = image.resize((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new('L', (28, 28), color=0)
    new_img.paste(image, (4, 4))
    new_img = new_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(new_img).unsqueeze(0)
    return tensor, new_img

# Main app
def main():
    st.title("MNIST Digit Recognition System")
    st.write("Draw a digit or upload an image. The model achieves >99% accuracy on MNIST.")

    with st.sidebar:
        st.header("Settings")
        conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.7)
        show_proc = st.checkbox("Show preprocessed image", True)
        # For uploaded images, allow inversion toggle
        invert_upload = st.checkbox("Invert uploaded image colors (if digit is dark on light)", True)

    model, device = get_model()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("✏️ Draw Here")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if st.button("Recognize Drawing", type="primary"):
            if canvas.image_data is not None:
                tensor, proc_img = preprocess_canvas(canvas)
                if tensor is None:
                    st.warning("No drawing detected. Please draw a digit.")
                else:
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        out = model(tensor)
                        probs = F.softmax(out, dim=1).cpu().numpy()[0]
                        conf, pred = torch.max(out, 1)
                        conf = torch.softmax(out, dim=1)[0, pred].item()
                    st.session_state['pred'] = pred.item()
                    st.session_state['conf'] = conf
                    st.session_state['probs'] = probs
                    st.session_state['proc_img'] = proc_img
                    st.session_state['source'] = 'draw'
                    st.rerun()

    with col2:
        if 'pred' in st.session_state:
            pred = st.session_state['pred']
            conf = st.session_state['conf']
            probs = st.session_state['probs']
            proc_img = st.session_state['proc_img']
            src = st.session_state['source']

            st.subheader("📊 Result")
            col_a, col_b = st.columns(2)
            col_a.metric("Predicted Digit", pred)
            col_b.metric("Confidence", f"{conf*100:.1f}%")

            if conf >= conf_thresh:
                st.success("✅ High confidence")
            else:
                st.warning("⚠️ Low confidence - try drawing clearer")

            if show_proc and proc_img:
                st.image(proc_img, caption="Preprocessed (28x28)", width=140)

            # Probability plot
            fig = go.Figure(data=[go.Bar(
                x=[str(i) for i in range(10)],
                y=probs,
                marker_color=['red' if i == pred else 'blue' for i in range(10)],
                text=[f'{p*100:.1f}%' for p in probs],
                textposition='outside'
            )])
            fig.update_layout(height=300, title="Probabilities", yaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

            if conf > 0.95:
                st.balloons()
        else:
            st.info("Draw a digit and click Recognize")

    # Upload section
    st.markdown("---")
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Choose an image", type=['png','jpg','jpeg','bmp'])
    if uploaded:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded)
        col1.image(img, caption="Original", width=200)
        if col2.button("Recognize Uploaded"):
            tensor, proc_img = preprocess_upload(img, invert=invert_upload)
            if tensor is None:
                st.warning("No digit detected in image.")
            else:
                tensor = tensor.to(device)
                with torch.no_grad():
                    out = model(tensor)
                    probs = F.softmax(out, dim=1).cpu().numpy()[0]
                    conf, pred = torch.max(out, 1)
                    conf = torch.softmax(out, dim=1)[0, pred].item()
                st.session_state['pred'] = pred.item()
                st.session_state['conf'] = conf
                st.session_state['probs'] = probs
                st.session_state['proc_img'] = proc_img
                st.session_state['source'] = 'upload'
                st.rerun()

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: gray;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    Made by <b>Dr. Mahroona Laraib</b> |
    <a href="https://github.com/drmahroona" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()