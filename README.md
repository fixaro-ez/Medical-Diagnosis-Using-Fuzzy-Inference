# Multi-Disease Fuzzy Expert System 🩺

A modular medical diagnostic application using Fuzzy Logic to assess risks across multiple disease domains. Built with Python, Streamlit, and Scikit-Fuzzy.

## 🌟 Project Overview
This system provides a unified interface for four distinct medical domains. By utilizing **Fuzzy Inference Systems (FIS)**, the application can handle the "gray areas" of medical symptoms better than traditional binary logic.

### Disease Modules:
* **Respiratory:** Logic focusing on SpO2, cough frequency, and breath shortness.
* **Heart:** Cardiovascular risk based on BP, cholesterol, and heart rate.
* **Diabetes:** Metabolic health using glucose levels and BMI.
* **Infectious Disease:** Assessment based on temperature, fatigue, and exposure.

---

## 🏗️ Technical Architecture
We use a **Decoupled Architecture**. The UI (`main.py`) acts as a "Renderer" while each folder in `modules/` acts as an independent "Logic Provider."


## 🚀 Setup & Installation

### 👨‍🏫 For General Users / Evaluators
1. **Clone the repository:**
   git clone https://github.com/fixaro-ez/Medical-Diagnosis-Using-Fuzzy-Inference.git

2. **Create a virtual environment:**
     python -m venv venv
3. **Switching to a virtual environment:**
    * .\venv\Scripts\Activate.ps1 (for powershell)
    * .\venv\Scripts\activate.bat (for command prompt)
    * source ./venv/Scripts/activate (for bash)

4. **Install Dependencies:**
    pip install -r requirements.txt

5. **Launch the Application:**
    streamlit run main.py    

### 🛠 Collaborator Workflow (For Team Members)
    
1. **Create a Feature Branch:**
   Never push directly to main. Create a branch for your specific disease:
 **git checkout -b [feature-name]**

2. **Daily Sync (Important):** 

 ##### While on your feature branch: **git pull origin main**


3. **Write Code on your branch**

4. **Push Changes to to your branch:**   
   **git push origin [feature-name]**