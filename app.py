
# ==========================================
# 2. IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import cv2
import torch
import whisper
import re
import os
import shutil
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from PIL import Image
import easyocr
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using Device: {device}")

MODEL_PATH = "./saved_fake_news_bert_model"

# ==========================================
# 3. ROBUST DATASET PREPARATION
# ==========================================
def prepare_dataset():
    print("🧹 Preparing and Cleaning Dataset...")
    
    df = None
    if os.path.exists("combined_news_data.csv"):
        print("   -> Found 'combined_news_data.csv', loading...")
        df = pd.read_csv("combined_news_data.csv")
    else:
        print("   -> Merging uploaded CSV files...")
        data_frames = []
        
        # Load True.csv
        if os.path.exists('True.csv'):
            t = pd.read_csv('True.csv')
            t['text'] = t['title'].fillna('') + " " + t['text'].fillna('')
            t['label'] = 1
            data_frames.append(t[['text', 'label']])
            
        # Load Fake.csv
        if os.path.exists('Fake.csv'):
            f = pd.read_csv('Fake.csv')
            f['text'] = f['title'].fillna('') + " " + f['text'].fillna('')
            f['label'] = 0
            data_frames.append(f[['text', 'label']])
            
        # Load fake_and_real_news.csv
        if os.path.exists('fake_and_real_news.csv'):
            fr = pd.read_csv('fake_and_real_news.csv')
            fr = fr.rename(columns={'Text': 'text'})
            fr['label'] = fr['label'].map({'Fake': 0, 'Real': 1})
            data_frames.append(fr[['text', 'label']])
            
        # Load fakenrealnews.csv
        if os.path.exists('fakenrealnews.csv'):
            frn = pd.read_csv('fakenrealnews.csv')
            frn['text'] = frn['title'].fillna('') + " " + frn['text'].fillna('')
            frn['label'] = frn['label'].map({'FAKE': 0, 'REAL': 1})
            data_frames.append(frn[['text', 'label']])
            
        if not data_frames:
            print("❌ No CSV files found. Creating dummy data.")
            df = pd.DataFrame({'text': ['test text'], 'label': [0]})
        else:
            df = pd.concat(data_frames, ignore_index=True)

    # STRICT LABEL CLEANING
    def enforce_integer_label(x):
        try:
            s = str(x).lower().strip()
            if s in ['fake', '0', '0.0', 'false']: return 0
            if s in ['real', '1', '1.0', 'true']: return 1
            return -1
        except: return -1

    df['label'] = df['label'].apply(enforce_integer_label)
    df = df[df['label'] != -1]
    df = df.dropna(subset=['text'])
    df['label'] = df['label'].astype(int)
    
    df.to_csv("combined_news_data.csv", index=False)
    print(f"✅ Dataset Ready: {len(df)} rows. (Using ALL Data)")
    return df

def text_cleaning(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 4. TRAINING FUNCTION (WITH GRAPH)
# ==========================================
def train_model():
    # 1. Get Data
    df = prepare_dataset()
    
    print("   -> Cleaning Text...")
    df['text'] = df['text'].apply(text_cleaning)
    
    # 2. Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.15
    )
    
    # 3. Tokenize
    print("⏳ Tokenizing Full Dataset...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    # 4. Dataset Class
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]) 
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    
    # 5. Initialize Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    # 6. Training Args (LIVE VALIDATION)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        
        # VISUAL SETTINGS
        logging_steps=50,       
        eval_strategy="steps",  
        eval_steps=100,         
        
        report_to="none",
        disable_tqdm=False, 
        save_strategy="no",
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print("🚀 Starting Training on FULL DATASET...")
    trainer.train()
    print("✅ Training Finished.")
    
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    # ------------------------------------
    # PLOT VALIDATION LOSS GRAPH
    # ------------------------------------
    print("📊 Generating Loss Graph...")
    log_history = trainer.state.log_history
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        elif 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.6)
    plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show() # Displays graph in Colab output
    
    return tokenizer, model

# ==========================================
# 5. EXECUTION BLOCK
# ==========================================
if os.path.exists(MODEL_PATH):
    print("✅ Found saved model. Loading...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    except:
        print("⚠️ Model corrupted. Retraining...")
        shutil.rmtree(MODEL_PATH, ignore_errors=True)
        tokenizer, model = train_model()
else:
    if os.path.exists('./results'): shutil.rmtree('./results')
    tokenizer, model = train_model()

# ==========================================
# 6. LOAD INFERENCE TOOLS
# ==========================================
print("🔄 Loading Whisper, Vision & OCR...")
whisper_model = whisper.load_model("base")
vision_pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model", device=0 if device=="cuda" else -1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
ocr_reader = easyocr.Reader(['en'], gpu=(device=='cuda'))

# ==========================================
# 7. AGGRESSIVE ANALYSIS LOGIC (PEAK DETECTION)
# ==========================================
def analyze_video(video_path):
    if not video_path: return "No video."
    
    print(f"🔍 Analyzing {video_path} in AGGRESSIVE MODE...")
    
    # --- A. VISUAL ANALYSIS ---
    cap = cv2.VideoCapture(video_path)
    fake_scores = []
    frames_checked = 0
    faces_found = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Check every 5th frame
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Lower minNeighbors to 3 for aggressive detection
                faces = face_cascade.detectMultiScale(gray, 1.1, 3)
                if len(faces) > 0:
                    faces_found += 1
                    for (x, y, w, h) in faces:
                        margin = int(w * 0.2)
                        x1, y1 = max(0, x - margin), max(0, y - margin)
                        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
                        face = frame[y1:y2, x1:x2]
                        pil_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        preds = vision_pipe(pil_img)
                        score = next((p['score'] for p in preds if p['label'] in ['Deepfake', 'Fake']), 0)
                        fake_scores.append(score)
                    frames_checked += 1
            except: pass
    cap.release()
    
    # PEAK DETECTION
    if fake_scores:
        peak_fake = max(fake_scores) * 100
        avg_fake = np.mean(fake_scores) * 100
        
        if peak_fake > 80:
            vis_verdict = "FAKE (Glitch Detected)"
            vis_confidence = peak_fake
        elif avg_fake > 45:
            vis_verdict = "FAKE (General)"
            vis_confidence = avg_fake
        else:
            vis_verdict = "REAL"
            vis_confidence = 100 - avg_fake
    else:
        vis_verdict = "UNCERTAIN"
        vis_confidence = 0

    # --- B. AUDIO/TEXT ANALYSIS ---
    try:
        transcription = whisper_model.transcribe(video_path)
        text_content = transcription["text"]
        inputs = tokenizer(text_cleaning(text_content), return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            text_fake = probs[0][0].item() * 100
        txt_verdict = "FAKE" if text_fake > 50 else "REAL"
    except:
        text_content = "Error or No Audio"
        text_fake = 0
        txt_verdict = "UNCERTAIN"

    # --- C. OCR ANALYSIS ---
    try:
        cap = cv2.VideoCapture(video_path)
        ocr_text = ""
        count = 0
        while cap.isOpened() and count < 3:
            ret, frame = cap.read()
            if not ret: break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 50 == 0:
                res = ocr_reader.readtext(frame, detail=0)
                ocr_text += " ".join(res) + " "
                count += 1
        cap.release()
    except: ocr_text = ""

    # --- FINAL VERDICT ---
    if "FAKE" in vis_verdict:
        final_verdict = "MANIPULATED VIDEO"
        final_score = vis_confidence
        color = "#e74c3c" # Red
    elif txt_verdict == "FAKE":
        final_verdict = "FAKE NEWS (Real Video)"
        final_score = text_fake
        color = "#e67e22" # Orange
    else:
        final_verdict = "AUTHENTIC"
        final_score = vis_confidence
        color = "#27ae60" # Green

    html = f"""
    <div style="background: white; padding: 20px; border-radius: 10px; border: 4px solid {color};">
        <h1 style="color: {color}; text-align: center;">{final_verdict}</h1>
        <h3 style="text-align: center; color: #555;">Confidence: {final_score:.1f}%</h3>
        <hr>
        <div style="display: flex;">
            <div style="flex: 1; padding: 10px;">
                <h3>👁️ Visual Analysis</h3>
                <p>Verdict: <b>{vis_verdict}</b></p>
                <p>Confidence: {vis_confidence:.1f}%</p>
                <p>Faces Checked: {faces_found}</p>
            </div>
            <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                <h3>🧠 Content Analysis</h3>
                <p>Verdict: <b>{txt_verdict}</b></p>
                <p>Confidence: {text_fake:.1f}%</p>
            </div>
        </div>
        <div style="background: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 5px;">
            <p><b>Transcript:</b> {text_content[:300]}...</p>
            <p><b>Screen Text:</b> {ocr_text[:200]}</p>
        </div>
    </div>
    """
    return html

# ==========================================
# 8. LAUNCH UI
# ==========================================
iface = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload News Video"),
    outputs=gr.HTML(label="Multimodal Report"),
    title="Deepfake & Fake News Detector (Full Production)",
    description="Full training on 55k+ rows. Aggressive Peak Detection for visual deepfakes. Validation Graph included."
)
iface.launch(share=True, debug=True)