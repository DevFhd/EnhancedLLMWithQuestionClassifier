Multi-Level Question Classification & Answer Generation
This project implements a multi-level question classification system using HuggingFace Transformers (DistilBERT) for classification, alongside the Ollama Phi model for answer generation. It supports training, evaluation, and deployment with an optional web interface.

Features
Multi-level question classification with fine-tuned DistilBERT models.

Answer generation powered by Ollama's Phi model.

Custom dataset support for flexible training.

Colab-friendly training scripts to leverage GPU resources.

Experimental web interface (Streamlit) under active development.

Project Structure
File / Folder	Description
train.py	Script to fine-tune DistilBERT models
app.py	Streamlit application for classification & Q&A
models/	Folder containing trained model checkpoints
requirements.txt	Python dependencies

Getting Started
Step 1: Install Dependencies
Install the required Python packages:
pip install -r requirements.txt
Step 2: Train the Models
Train the DistilBERT classification models:

Open train.py in Google Colab (recommended with T4 GPU for performance).

Upload your dataset file (e.g., Question_Classification_Dataset.csv).

Run the training script:
python train.py
Models will be saved to the specified directory (e.g., models/model_cat0, models/model_cat1, etc.).

Step 3: Run the Prediction Pipeline
Unzip trained models if provided as .zip files.

Edit model paths in app.py to point to your local models:
tokenizer_cat, model_cat = load_model("models/model_cat0")
Launch the Streamlit app:
streamlit run app.py
Step 4: Enable Answer Generation (Ollama Phi Model)
To use the Q&A feature powered by Ollama Phi:

Install Ollama from https://ollama.com.

Pull the Phi model:
ollama pull phi
Start the Ollama service and ensure it is running before running the Streamlit app.

Example Output
Input:
Who discovered gravity?

Output:

Category0: 0 (HUMAN)
Category1: (ind)
Answer: Isaac Newton discovered gravity in the 17th century.
Author
Developed by Fahad Alhajji as an open-source NLP & AI project.

Note: Paths in scripts must be updated to match your local file structure. The web interface is under development and may require path adjustments.
