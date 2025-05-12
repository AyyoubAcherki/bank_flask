from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle (assurez-vous que le modèle 'ch.joblib' est dans le bon dossier)
model = joblib.load('C:\\Users\\Probook\\Desktop\\python\\ML\\bank\\model.joblib')

# Route de la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour effectuer la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Récupération des valeurs du formulaire
    age = int(data['age'])
    balance = float(data['balance'])
    duration = int(data['duration'])
    campaign = int(data['campaign'])
    previous = int(data['previous'])
    
    # Récupération des valeurs des options radio
    default = int(data['default'])
    housing = int(data['housing'])
    loan = int(data['loan'])
    
    # Récupération des valeurs de l'éducation
    education_primary = int(data.get('education_primary', 0))
    education_secondary = int(data.get('education_secondary', 0))
    education_tertiary = int(data.get('education_tertiary', 0))
    education_unknown = int(data.get('education_unknown', 0))
    
    # Récupération du statut marital
    marital_married = int(data.get('marital_married', 0))
    marital_single = int(data.get('marital_single', 0))
    marital_divorced = int(data.get('marital_divorced', 0))
    
    # Récupération de la profession
    job_admin = int(data.get('job_admin', 0))
    job_blue_collar = int(data.get('job_blue-collar', 0))
    job_entrepreneur = int(data.get('job_entrepreneur', 0))
    job_housemaid = int(data.get('job_housemaid', 0))
    job_management = int(data.get('job_management', 0))
    job_retired = int(data.get('job_retired', 0))
    job_self_employed = int(data.get('job_self-employed', 0))
    job_services = int(data.get('job_services', 0))
    job_student = int(data.get('job_student', 0))
    job_technician = int(data.get('job_technician', 0))
    job_unemployed = int(data.get('job_unemployed', 0))
    job_unknown = int(data.get('job_unknown', 0))
    
    # Création du vecteur de features pour la prédiction
    features = np.array([[age, balance, duration, campaign, previous, 
                          default, housing, loan, 
                          education_primary, education_secondary, education_tertiary, education_unknown,
                          marital_married, marital_single, marital_divorced,
                          job_admin, job_blue_collar, job_entrepreneur, job_housemaid, job_management, job_retired,
                          job_self_employed, job_services, job_student, job_technician, job_unemployed, job_unknown]])

    # Prédiction
    prediction = model.predict(features)[0]
    result = "Client à risque de départ" if prediction == 1 else "Client fidèle"
    
    return render_template('index.html', prediction=result)

# Démarrage de l'application
if __name__ == '__main__':
    app.run(debug=True)
