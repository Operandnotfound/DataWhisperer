import gradio as gr
import pandas as pd
import nltk
from nltk.corpus import wordnet
import random
import os
import tempfile
from PIL import Image

# =================== Setup =====================

# Optional YOLO (free pre-trained model for image tasks)
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # lightweight model
except Exception as e:
    print("⚠️ YOLO not available:", e)
    model = None

# Download NLTK resources (first time only)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# =================== Functions =====================

# 1. CSV Cleaner
def clean_csv(file):
    try:
        df = pd.read_csv(file)

        # Drop duplicates + nulls
        df = df.drop_duplicates().dropna()

        # Save cleaned file in temp dir
        tmp_dir = tempfile.mkdtemp()
        cleaned_file = os.path.join(tmp_dir, "cleaned_dataset.csv")
        df.to_csv(cleaned_file, index=False)

        return cleaned_file
    except Exception as e:
        return f"⚠️ Error while cleaning CSV: {e}"

# 2. Text Augmenter
def augment_text(text):
    if not text.strip():
        return "⚠️ Please enter some text."

    stopwords = {"the","is","a","an","in","on","of","and","to","for","it","this"}
    words = text.split()
    new_words = []

    for w in words:
        if w.lower() in stopwords:
            new_words.append(w)
            continue
        syns = wordnet.synsets(w)
        if syns and random.random() > 0.7:
            lemmas = syns[0].lemma_names()
            if lemmas:
                choice = random.choice(lemmas).replace("_", " ")
                new_words.append(choice)
                continue
        new_words.append(w)

    return " ".join(new_words)

# 3. Image Labeler
def process_image(img_path):
    try:
        if model:
            results = model(img_path)
            annotated_img = results[0].plot()
            return Image.fromarray(annotated_img)
        else:
            return Image.open(img_path)  # fallback: return original
    except Exception as e:
        return f"⚠️ Error processing image: {e}"

# =================== Gradio UI =====================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌟 AI Data Whisperer  
        Open-source **dataset cleaner, augmenter & labeler** (zero-cost).  
        ✅ No API keys needed | 🪶 Lightweight and Open-Source | ⚡ Built By Ibrahim Ali  
        """
    )

    with gr.Tab("📊 CSV Cleaner"):
        with gr.Row():
            file_in = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
            file_out = gr.File(label="Download Cleaned CSV", type="filepath")
        file_btn = gr.Button("🚀 Clean CSV")
        file_btn.click(clean_csv, inputs=file_in, outputs=file_out)

    with gr.Tab("📝 Text Augmenter"):
        txt_in = gr.Textbox(label="Enter text", placeholder="Type a sentence...")
        txt_btn = gr.Button("✨ Augment Text")
        txt_out = gr.Textbox(label="Augmented text")
        txt_btn.click(augment_text, inputs=txt_in, outputs=txt_out)

    with gr.Tab("🖼️ Image Labeler"):
        img_in = gr.Image(label="Upload image", type="filepath")
        img_btn = gr.Button("🔍 Label Objects")
        img_out = gr.Image(label="Labeled image", type="pil")
        img_btn.click(process_image, inputs=img_in, outputs=img_out)

demo.launch(inbrowser=True, share=False)
