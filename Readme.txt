Team name : Watt 

# ⚡ VidyutAi-Hackathon

> **EV-AI — Intelligent Energy Solutions for the Future**

This repository contains the code, datasets, and notebooks created for the **VidyutAI Hackathon** project.  
It demonstrates how data-driven AI models can help optimize Electric Vehicle (EV) systems — including data analysis, model development, and interactive dashboards.

QUICK SETUP:
- THE model-save folder contains joblib files for the model trained you just need to clone the repo and change the path for the directory in app.py file 
- after changing the directory path in local machine simply run the app.py if it says "Running on http://127.0.0.1:5000" it is succesfully running
- at the end just run the code "streamlit run dashboard2.py" and you can run the dashboard smoothly! :)
---

## 📘 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Notebooks](#notebooks)
- [Applications](#applications)
- [Datasets](#datasets)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🚀 Overview

**VidyutAI-Hackathon** aims to leverage machine learning and AI for smarter energy and EV analytics.  
The project includes:
- **Data creation & preprocessing** (`DataCreator.ipynb`)
- **Model training & evaluation** (`Main.ipynb`)
- **Multiple interactive apps & dashboards** (`app.py`, `app2.py`, etc.)
- **Supporting datasets** for experimentation

The focus is on applying data science to real-world EV or energy-related challenges — such as efficiency prediction, performance analysis, or intelligent monitoring.

---

## 📂 Repository Structure



.
├── Data.csv
├── Full-Data.csv
├── data2.csv
├── DataCreator.ipynb # Dataset generation & preprocessing notebook
├── Main.ipynb # Main model training & analysis notebook
├── app.py # Streamlit/Flask/Dash app (demo 1)
├── app2.py # Alternate demo app
├── app3.py # Alternate demo app
└── dashboard2.py.py # Dashboard script (rename to dashboard2.py recommended)


---

## ⚙️ Quickstart

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/anujjtiwari/VidyutAi-Hackathon.git
cd VidyutAi-Hackathon

2️⃣ Set Up a Virtual Environment (Recommended)
python -m venv .venv
# Activate:
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

3️⃣ Install Dependencies

(A requirements.txt file can be added — see suggestions below)

Example packages commonly used:

pip install numpy pandas scikit-learn matplotlib jupyter
pip install streamlit flask dash plotly

4️⃣ Run the Notebooks
jupyter notebook
# Then open DataCreator.ipynb or Main.ipynb in the browser

📓 Notebooks
DataCreator.ipynb

Prepares and cleans datasets (Data.csv, Full-Data.csv, etc.)

Handles preprocessing like normalization, feature generation, and merging.

Main.ipynb

Main analysis pipeline:

Exploratory Data Analysis (EDA)

Model training & testing

Result visualization

🧠 Applications
app.py, app2.py, app3.py

Interactive dashboards or web apps (likely Streamlit, Flask, or Dash based)

Used to visualize data insights and model outputs.

dashboard2.py.py

Visual dashboard (consider renaming to dashboard2.py).

Likely integrates multiple views or charts for the dataset/model.

To run:
# If Streamlit:
streamlit run app.py

# If Flask or Dash:
python app.py


Then open the local URL displayed in your terminal (usually http://127.0.0.1:5000/).

📊 Datasets
File	Description
Data.csv	Processed dataset for initial model training
data2.csv	Extended dataset with new parameters
Full-Data.csv	Final combined dataset used for model testing

Use DataCreator.ipynb to regenerate or customize these datasets.

🧩 Usage

Open DataCreator.ipynb → generate datasets

Open Main.ipynb → train models and visualize performance

Run app.py (or app2.py, app3.py) → launch dashboard demo

🔮 Future Enhancements

 Add requirements.txt with exact package versions

 Merge dashboards into a single unified web app

 Add deployment via Streamlit Cloud or HuggingFace Spaces

 Include saved model files (.pkl / .onnx) for quick loading

 Add visual report (accuracy plots, confusion matrix, etc.)

 Rename dashboard2.py.py → dashboard2.py

🤝 Contributing

Contributions are welcome!

Fork this repository

Create your feature branch:

git checkout -b feat/your-feature


Commit your changes and push

Submit a Pull Request with a clear description

🪪 License

This project currently does not include a license.
To make it open-source, add a LICENSE file (e.g., MIT, Apache-2.0, or GPL-3.0).

Example MIT license header:

MIT License
Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy...

📬 Contact

Maintainer: @anujjtiwari

For collaborations or clarifications, open an Issue
.
        
