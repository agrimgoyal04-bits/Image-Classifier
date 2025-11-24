🖼️ Image Classifier using Lightweight Vision Transformers

  A complete deep learning project for image classification, built using DeiT-Tiny and ViT-Tiny Vision Transformers.

The project includes full training scripts, dataset preprocessing tools, inference notebooks, a Flask-based web application, and four trained model weights:

  1. DeiT-Tiny (trained on 600-image custom dataset)
  2. DeiT-Tiny (trained on pooled dataset)
  3. ViT-Tiny (trained on 600-image custom dataset)
  4. ViT-Tiny (trained on pooled dataset)
     
This repository is designed to showcase how dataset size, diversity, and model architecture affect performance and generalization.


📁 Repository Structure
├── data/
│   ├── images/                  # 600-image custom dataset
│   ├── pooled_images/           # Larger pooled dataset
│   ├── labels.csv               # Annotations for custom dataset
│   ├── processed_dataset.csv    # Cleaned dataset
│   ├── attributes.yaml          # Attribute schema
│   ├── classes.txt              # Class labels (10 classes)
│   └── ...  
│
├── other_codes/
│   ├── clean_attributes.py
│   ├── remove_synonyms.py
│   └── split.py                 # Train/val split utility
│
├── src/
│   ├── dataset.py               # Custom dataset loader
│   ├── dataset_pooled.py        # Loader for pooled dataset
│   ├── model.py                 # DeiT-Tiny & ViT-Tiny architectures
│   ├── model_pooled.py
│   ├── retrieval.py             # Cosine similarity retrieval
│   ├── train_deit_tiny.py
│   ├── train_deit_tiny_pooled.py
│   ├── train_vit_tiny.py
│   └── train_vit_tiny_pooled.py
│
├── outputs/
│   ├── best_model.pth           # DeiT-Tiny (custom dataset)
│
├── outputs_pooled/
│   ├── best_model.pth           # DeiT-Tiny (pooled dataset)
│
├── outputs_vit_tiny/
│   ├── best_model.pth           # ViT-Tiny (custom dataset)
│
├── outputs_vit_tiny_pooled/
│   ├── best_model.pth           # ViT-Tiny (pooled dataset)
│
├── web_app/
│   ├── app.py                   # Flask web application
│   ├── templates/               # HTML templates
│   └── uploads/                 # Uploaded images during inference
│
├── inference_notebooks/
│   ├── deit_tiny_inference.ipynb
│   └── vit_tiny_inference.ipynb
│
└── README.md


📦 Features

✔ Two Lightweight Vision Transformers : 
  1. DeiT-Tiny (distilled: stable & efficient)
  2. ViT-Tiny (pure transformer baseline)
     
✔ Dual-Dataset Training :
  Small 600-image custom dataset
  Large pooled dataset for generalization comparison

✔ Complete Training Pipeline :
  1. Dataset loading
  2. Augmentation
  3. Fine-tuning
  4. Validation
  5. Saving best checkpoint

✔ Four Trained Weights Included :
  1. DeiT-Tiny — Custom Dataset
  2. DeiT-Tiny — Pooled Dataset
  3. ViT-Tiny — Custom Dataset
  4. ViT-Tiny — Pooled Dataset

✔ Flask Web App : Upload an image → Model predicts its class + attributes.

✔ Inference Notebooks : Simple .ipynb notebooks for loading .pth weights and running predictions.

🧠 Model Training Details : 

- Framework: PyTorch + timm
- Optimizer: AdamW
- Loss: Cross-Entropy
- Scheduler: Cosine Annealing
- Batch Size: 32
- Epochs: ~20–30
- Augmentations:
  1. Random Resized Crop
  2. Color Jitter
  3. Horizontal Flip
- Identical hyperparameters were used across both datasets to ensure fair comparison.
  
📈 Evaluation Metrics

The following metrics were used:

      Metric	                 Why
1. Accuracy	            Overall correctness
2. Precision/Recall	    Understand FP/FN behavior
3. Macro F1	            Equal weight to all classes
4. Micro F1	            Weighted by class frequency
5. Confusion Matrix	    Shows misclassified classes

   
🔍 Key Results Summary

DeiT-Tiny
- Better performance on small dataset
- Stable training due to distillation
- Higher F1 scores on custom dataset

ViT-Tiny
- Strong improvement on pooled dataset
- Requires more data compared to DeiT-Tiny
- Better generalization with diverse samples

Common Challenges
- Visually similar classes (e.g., mug vs jug)
- Lighting variations
- Transparent/glass objects harder to identify
  
🚀 Running the Web App

Install dependencies:
pip install -r requirements.txt

Start the Flask app:
python web_app/app.py

Then open the local URL shown in terminal:
http://127.0.0.1:5000/
Upload an image → Model predicts class and attributes.

🔬 Running Inference Manually
Use any of the notebooks:

inference_notebooks/deit_tiny_inference.ipynb
inference_notebooks/vit_tiny_inference.ipynb

Each notebook includes:
- Code to load the architecture
- Load desired .pth weight
- Preprocess input
- Output prediction

🌍 Applications

1. Everyday object recognition
2. E-commerce auto-tagging
3. Smart inventory classification
4. Visual assistance tools
5. Lightweight edge-device deployment

🔮 Future Improvements

- Add CLIP-based multimodal retrieval
- Integrate attribute prediction
- Improve dataset diversity
- Convert model to ONNX / TFLite for mobile
- Add visualization dashboard
