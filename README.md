#  Factory Optimization Project

This project applies **Machine Learning + Optimization** to predict shipping lead times and support factory reallocation decisions.  
It includes a **Streamlit dashboard** for interactive analysis and visualization.

---

##  Features
- Predicts **Lead Time (days)** using Gradient Boosting Regression.
- Cleans and processes distributor order data.
- Maps products to factories for optimization insights.
- Interactive **Streamlit dashboard** for predictions and visualizations.
- Saved artifacts for reproducibility:
  - `model.pkl` → trained ML model
  - `feature_columns.pkl` → encoded feature list
  - `cleaned_data.csv` → cleaned dataset

---

##  Project Structure
factory_optimization_project/
│
├── data/                         # Raw CSV files
├── src/                          # Source code
│   ├── data_processor.py          # Training script
│   ├── dashboard.py               # Streamlit app
│   └── visualization.py           # Plots & charts
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

---

##  Installation
Clone the repository:
```bash
git clone https://github.com/mythrivh/my_project.git
cd my_project

Install dependencies:
pip install -r requirements.txt

Usage
1. Train the Model
Run the training script:
python src/data_processor.py
This will generate:
model.pkl
feature_columns.pkl
cleaned_data.csv

2. Launch the Dashboard
Start Streamlit:
py -m streamlit run src/app.py
Open the link in your browser to interact with the app.

Example Outputs
Model Performance: MAE ≈ 2–3 days, R² ≈ 0.8 (depending on dataset).
Dashboard: Predict lead times by selecting product, factory, region, etc.

 Notes
Ensure your dataset (Nassau Candy Distributor.csv) is placed in the data/ folder.
Update requirements.txt if you add new libraries.
Contributions and suggestions are welcome

Author
Mythri V H  
Aspiring Web Developer & ML Intern
