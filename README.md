# 🦴 OncoAid: Bone Tumor Treatment Prediction System

OncoAid is a web-based application that leverages machine learning (ML), deep learning (DL), and large language models (LLMs) to predict treatment options for bone cancer based on patient data. It also provides preventive healthcare tips using integrated LLMs.

## 🚀 Features

- Predict bone cancer treatments using trained ML/DL models
- Quantum ML and QNN-based model experimentation
- Gemini LLM integration for prevention & safety tips
- User authentication system (signup/login)
- Responsive frontend with intuitive UI
- Dockerized for easy deployment

## 🧠 Technology Stack

### ⚙️ Backend
- Python
- Flask
- ML/DL Models (Scikit-learn, TensorFlow/Keras)
- Quantum ML (Qiskit, Pennylane) *(legacy component)*

### 🖼️ Frontend
- HTML, CSS, JavaScript
- Bootstrap
- Jinja2 Templates

### 🔐 DevOps & Deployment
- Docker
- Docker Hub ([parinitha377](https://hub.docker.com/u/parinitha377))

### 🧠 LLM Integration
- Gemini API for prevention tips

## 🏥 Input Parameters

- **Age** (numeric)
- **Sex** (categorical): Female, Male
- **Grade** (categorical): High, Intermediate
- **Histological Type** (categorical): 13 types
- **MSKCC Type** (categorical): Leiomyosarcoma, MFH, Synovial sarcoma
- **Site of Primary STS** (categorical): 7 location types

## 🧪 Testing Strategy

Extensive testing was performed including:

- Unit Testing
- Integration Testing
- Sanity & Smoke Testing
- Security Testing
- Performance & Load Testing
- System & Acceptance Testing

Managed by **Desu Sree Vardhan** (Product Manager & Tester)

## 👥 Scrum Team

- **Scrum Master:** Nithin H  
- **Product Manager & Tester:** Desu Sree Vardhan  
- **ML/DL/QML/QNN Integration:** Vaishnav  
- **Frontend, DevOps & LLM Integration:** Parinitha RK  

## 🧾 Installation & Usage

### Prerequisites
- Python 3.8+
- Docker (optional for containerized deployment)

### Clone the Repo
```bash
git clone https://github.com/yourusername/oncoaid.git
cd oncoaid
Setup Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run Locally
bash
Copy
Edit
python app.py
Docker Deployment
bash
Copy
Edit
docker build -t oncoaid .
docker run -p 5000:5000 oncoaid
📈 Future Developments
✅ Enhanced LLM response personalization using user history

✅ Role-based access for doctors, patients, and admins

✅ Integration with hospital EMR systems (FHIR support)

✅ REST API endpoints for mobile & third-party integration

✅ Deployment on cloud platforms (GCP/AWS with CI/CD)

✅ Model explainability using SHAP/LIME

✅ Feedback loop to retrain ML models from real-world input

✅ Localization & multilingual support for LLM tips

✅ Accessibility enhancements for visually impaired users

📬 Contact
For feedback, suggestions, or collaborations:

Parinitha RK
Frontend, DevOps & LLM Integration
Docker Hub: parinitha377

⚠️ This project is for educational and research purposes only. It is not intended for clinical use without professional validation.

yaml
Copy
Edit

---

Let me know if you want this in a downloadable `.md` file, or want to tailor it to a specific deployment platform like Heroku, AWS, or GCP.







