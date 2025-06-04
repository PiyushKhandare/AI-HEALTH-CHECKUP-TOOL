# AI HEALTH CHECKUP TOOL
 AI Health Checkup Tool is an intelligent web-based system designed to analyze user-provided health parameters (like age, symptoms, blood pressure, glucose levels, etc.) and predict potential health conditions using machine learning models. It offers instant health insights, disease risk predictions (e.g., diabetes, Alzheimer's, PCOD), and personalized health tips—empowering users to take early action and make informed medical decisions from the comfort of their homes.

## 🗂️ Project Structure
project_root/
│
├── alzheimer/ # Alzheimer's prediction module
├── bmi/ # BMI calculator logic
├── diabetes/ # Diabetes prediction module
├── kyd/ # Know Your Disease (symptom-based)
├── liver/ # Liver disease prediction module
├── models/ # Pre-trained ML models
├── static/ # CSS, JS, images, etc.
├── templates/ # HTML templates
│
├── app.py # Main Flask app
├── check_db.py # DB connection checker
├── appointments.db # SQLite database for appointments

# Database
Download SQLite database for storing the appointments data in the database.

# Libraries
pip install flask pandas scikit-learn numpy joblib

# How to Run
Clone or download the project.

Navigate to the project folder:

```bash
cd project_folder

Run the app:

python app.py

Open in browser:


http://127.0.0.1:5000/



