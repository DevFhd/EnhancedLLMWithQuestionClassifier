# **Multi-Level Question Classification & Answer Generation**

This project implements a **multi-level question classification system** using **HuggingFace Transformers (DistilBERT)** for classification and **Ollama Phi model** for answer generation. It supports training, evaluation, and deployment with optional web interface.

---

## ✅ Features
- **Multi-level classification** using fine-tuned DistilBERT.
- **Answer generation** using Ollama's Phi model.
- **Custom dataset support** for flexible training.
- **Colab-friendly training** for resource-heavy operations.
- Includes an **experimental web interface** (under development).

---

## 📦 Project Contents
- **Training Script**
  - Suggested to run on **Google Colab with T4 GPU** for better performance.
  - Includes dataset loading and model fine-tuning logic.
- **Running Script**
  - Loads trained models (provided as `.zip` or generated after training).
  - contains simple streamlit UI
  - Supports prediction and Q&A pipeline.
- **Web Interface (in progress)**
  - Simple front-end for user interaction.
- **Important:** You **must edit file paths** in the scripts according to your local project structure.

---

## 🚀 Getting Started

### 1. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
2. Training the Models
Run the training script to fine-tune the models:

Open train.py on Google Colab.

Upload your dataset file (e.g., Question_Classification_Dataset.csv).

Execute:

python train.py
Models will be saved to the specified directory (e.g., models/model_cat0, models/model_cat1, etc.).

3. Running the Prediction Pipeline
Unzip the trained models if they are provided as .zip files.

Edit the paths in the script to point to your models, for example:

tokenizer_cat, model_cat = load_model("models/model_cat0")
Then run:

python app.py
4. Enable Answer Generation (Ollama Phi Model)
To enable the Q&A feature:

Install Ollama from https://ollama.com.

Pull the Phi model:

ollama pull phi
Start Ollama service and make sure it's running before executing the pipeline.

⚠ Important Notes
Edit all paths to match your local file arrangement.

The web interface is still in development; paths may require adjustment.

Ollama is required for answer generation.

✅ Example Output
Input:

Who discovered gravity?
Output:

Category0: 0 (HUMAN)
Category1: (ind)
Answer: Isaac Newton discovered gravity in the 17th century.

Author
Developed by Fahad Alhajji as an open-source NLP & AI project.
