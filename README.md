# Medical Diagnosis Using Fuzzy Inference

A modular medical diagnostic application using **Fuzzy Logic** to assess health risks across multiple disease domains. Built with Python, Streamlit, and scikit-fuzzy.

## Overview

This system provides a unified interface for four distinct medical domains. By utilizing **Fuzzy Inference Systems (FIS)**, the application handles the inherent uncertainty in medical symptoms better than traditional binary logic—capturing the "gray areas" of real-world diagnosis.

## Disease Modules

| Module          | Inputs                                                                 | Output                                                         |
| --------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Heart**       | Age, Systolic BP, Cholesterol, Heart Rate, Smoking, Family History     | Cardiovascular risk score                                      |
| **Diabetes**    | Glucose, BMI, Age, Insulin, Diabetes Pedigree                          | Diabetes risk score                                            |
| **Respiratory** | Age + 9 symptom severity scores (cough, breathlessness, fatigue, etc.) | Disease classification (Asthma, TB, Pneumonia)                 |
| **Infectious**  | Age, Temperature (°F), Duration, Cough, Diarrhea, Breathing Difficulty | Condition classification (Viral Fever, Flu, Gastro, High-Risk) |

Each module uses CSV-calibrated fuzzy membership functions derived from real medical datasets.

## Project Structure

```
Medical-Diagnosis-Using-Fuzzy-Inference/
├── main.py                    # Streamlit UI application
├── requirements.txt           # Python dependencies
├── modules/
│   ├── template_engine.py     # Shared engine template
│   ├── heart/
│   │   ├── engine.py          # Heart disease fuzzy logic
│   │   └── data/heart.csv     # Calibration dataset
│   ├── diabetes/
│   │   ├── engine.py          # Diabetes fuzzy logic
│   │   └── data/diabetes.csv  # Calibration dataset
│   ├── respiratory/
│   │   ├── engine.py          # Respiratory disease fuzzy logic
│   │   └── respiratory symptoms and treatment.csv
│   └── infectious/
│       ├── engine.py          # Infectious disease fuzzy logic
│       └── data/infectious.csv# Calibration dataset
└── README.md
```

## Technical Architecture

The application uses a **Decoupled Architecture**:

- **UI Layer** (`main.py`): Streamlit-based renderer that dynamically builds input forms based on each module's `get_inputs()` definition
- **Logic Providers** (`modules/*/engine.py`): Independent fuzzy inference engines exposing:
  - `get_inputs()` → List of input field definitions
  - `run_inference(user_data)` → Risk assessment results

This design allows adding new disease modules without modifying the main application.

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/fixaro-ez/Medical-Diagnosis-Using-Fuzzy-Inference.git
cd Medical-Diagnosis-Using-Fuzzy-Inference

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

The app will open at `http://localhost:8501`

## Usage

1. Log in with a doctor/admin account
2. Select a disease module from the Diagnosis page
3. Enter patient profile and symptom inputs, then run diagnosis
4. Review Explanation (rule trace), Input Impact, and Comparison tabs
5. Save the case to history and reopen it later from Case History

## Added Clinical Workflow Features

- **Doctor Authentication** using `streamlit-authenticator`
- **Case Management** with persistent save/load to CSV storage
- **Comparative Analysis** against historical anonymized averages
- **Rule Tracing** showing strongest activated fuzzy rules per diagnosis
- **Plain-Language Summary** for each diagnosis result
- **Patient Profile View** with diagnosis timeline and editable doctor notes
- **Admin Dashboard** with aggregate anonymized system metrics

## Login and Storage

On first run, the app creates `app_data/auth_config.yaml` and `app_data/cases.csv`.

Default users:

- **Admin:** username `admin`, password `admin123`
- **Doctor:** username `doctor`, password `doctor123`

Change these credentials in `app_data/auth_config.yaml` after first launch.

## Dependencies

- **streamlit** – Web UI framework
- **scikit-fuzzy** – Fuzzy logic library
- **pandas** – Data manipulation
- **numpy** – Numerical computing
- **altair** – Visualization

## Contributing

1. Create a feature branch: `git checkout -b feature-name`
2. Sync with main: `git pull origin main`
3. Make changes and test
4. Push to your branch: `git push origin feature-name`
5. Open a Pull Request

## License

See [LICENSE](LICENSE) for details.
