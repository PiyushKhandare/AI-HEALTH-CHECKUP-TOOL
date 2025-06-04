
# 🧠 AI Health Checkup Tool

> Smart Flask‑based web app that uses machine‑learning models to predict disease risk, calculate BMI, and let users book doctor appointments—all from a single dashboard.

---

## 🗂️ Project Structure

```plaintext
project_root/
│
├── alzheimer/          # Alzheimer's prediction module
├── bmi/                # BMI calculator logic
├── diabetes/           # Diabetes prediction module
├── kyd/                # Know Your Disease (symptom-based)
├── liver/              # Liver disease prediction module
├── models/             # Pre‑trained ML model files (.pkl/.joblib)
├── static/             # CSS, JS, images
├── templates/          # Jinja2 HTML templates
│
├── app.py              # Main Flask application
├── check_db.py         # Quick script to verify DB connection
├── appointments.db     # SQLite database
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ✨ Features

- **Multi‑disease prediction**: Diabetes, Alzheimer’s, Liver disease, PCOD & more  
- **BMI calculator & KYD** (Know‑Your‑Disease symptom module)  
- **Appointment booking** with e‑mail confirmation  
- **Clean Bootstrap‑styled UI**, voice‑input ready  
- Modular codebase—each disease lives in its own folder, sharing a common Flask backend  

---

## ⚙️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/your‑username/ai‑health‑checkup.git
cd ai‑health‑checkup
```

2. **Create a virtual environment** (optional but recommended)

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install the core stack:

```bash
pip install flask pandas scikit-learn numpy joblib
```

---

## 🚀 Running Locally

1. **Start the Flask server**

```bash
python app.py
```

2. **Open in browser**

```url
http://127.0.0.1:5000/
```

---

### 🔧 Optional Environment Variables

| Variable     | Purpose                           | Default      |
|--------------|-----------------------------------|--------------|
| FLASK_ENV    | `development` enables hot‑reload  | production   |
| SECRET_KEY   | Session security                  | change‑me    |

---

## 📦 Model Files

Place all trained `.pkl` / `.h5` models inside **models/**. Check each module’s loader function for expected filenames.

---

## 🤝 Contributing

Pull requests are welcome! Open an issue first to discuss major changes.

---

## 📝 License

MIT © 2025 Piyush
