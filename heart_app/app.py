from flask import Flask, request, jsonify, render_template
import pickle, numpy as np, pandas as pd, os

# 1. BULLETPROOF ABSOLUTE PATH RESOLUTION
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.getcwd())

# Step "outside" heart_app into the main Cardio_assessment folder
PARENT_DIR = os.path.dirname(BASE_DIR)

# 2. FORCE FLASK TO LOOK IN THE MAIN FOLDER FOR TEMPLATES AND PLOTS
app = Flask(__name__, 
            template_folder=os.path.join(PARENT_DIR, 'templates'),
            static_folder=os.path.join(PARENT_DIR, 'plots'),
            static_url_path='/plots')

# Register the enumerate filter to fix the Jinja2 error
app.jinja_env.filters['enumerate'] = enumerate

# Force it to look in the main folder for models too
MDL_DIR = os.path.join(PARENT_DIR, 'models')

# 3. SAFE LOADING FUNCTION WITH DETAILED ERRORS
def load(filename): 
    filepath = os.path.join(MDL_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\n[!] ERROR: Cannot find '{filepath}'.\nCheck your 'models' folder to ensure the file is named EXACTLY '{filename}'.")
    return pickle.load(open(filepath, 'rb'))

# Load preprocessing artifacts
scaler       = load('scaler.pkl')
imputer      = load('imputer.pkl')
scaler_cols  = load('scaler_cols.pkl')
feat_names   = load('feature_names.pkl')

# Load machine learning models 
MODELS = {
    'Random Forest':       load('rf_model.pkl'), 
    'Logistic Regression': load('logistic_model.pkl'),
    'XGBoost (GBM)':       load('xgb_model.pkl'),
}

# ─── CLINICAL METADATA ────────────────────────────────────────────────────────
FEATURE_META = {
    'age':      {'label':'Age','unit':'years','type':'number','min':20,'max':80,'step':1,'placeholder':'e.g. 52','default':52},
    'gender':   {'label':'Gender','type':'select','options':{0:'Female',1:'Male'},'default':1},
    'cp':       {'label':'Chest Pain Type','type':'select','options':{0:'Typical Angina',1:'Atypical Angina',2:'Non-Anginal Pain',3:'Asymptomatic'},'default':0},
    'trestbps': {'label':'Resting Blood Pressure','unit':'mmHg','type':'number','min':80,'max':220,'default':120},
    'chol':     {'label':'Cholesterol','unit':'mg/dl','type':'number','min':100,'max':400,'default':200},
    'thalach':  {'label':'Max Heart Rate','unit':'bpm','type':'number','min':60,'max':220,'default':150},
    'exang':    {'label':'Exercise Induced Angina','type':'select','options':{0:'No',1:'Yes'},'default':0},
    'oldpeak':  {'label':'ST Depression (Oldpeak)','unit':'mm','type':'number','min':0.0,'max':6.0,'step':0.1,'default':0.0}
}

HIDDEN_DEFAULTS = {
    'fbs': 0.0,      
    'restecg': 1.0,  
    'slope': 1.0,    
    'ca': 0.0,       
    'thal': 2.0      
}

def risk_tier(prob):
    if prob < 0.3:
        return (1, 'Low Risk', 'text-emerald-400', 'Routine check-ups advised. Maintain current lifestyle.')
    elif prob < 0.7:
        return (2, 'Moderate Risk', 'text-yellow-400', 'Elevated markers present. Consultation recommended.')
    else:
        return (3, 'High Risk', 'text-rose-500', 'Immediate cardiovascular evaluation required.')

def clinical_analysis(raw, pred):
    flags = []
    tips = []
    
    if float(raw.get('trestbps', 120)) > 140:
        flags.append({'level': 'moderate', 'icon': '⚠️', 'label': 'Elevated Blood Pressure', 'detail': 'Resting BP > 140 mmHg'})
    if float(raw.get('chol', 200)) > 240:
        flags.append({'level': 'moderate', 'icon': '⚠️', 'label': 'High Cholesterol', 'detail': 'Serum cholesterol > 240 mg/dl'})
    if float(raw.get('oldpeak', 0.0)) > 2.0:
        flags.append({'level': 'high', 'icon': '🚨', 'label': 'ST Depression', 'detail': 'Significant ST depression detected'})
    
    if pred == 1:
        tips.append({'icon': '🩺', 'text': 'Schedule a consultation with a cardiologist.'})
        tips.append({'icon': '📈', 'text': 'Monitor blood pressure and cholesterol strictly.'})
    else:
        tips.append({'icon': '🏃', 'text': 'Maintain a healthy diet and regular cardiovascular exercise.'})
    
    return flags, tips

@app.route('/')
def home():
    # ─── UPDATE THESE 6 NAMES TO EXACTLY MATCH YOUR MAC FOLDER ───
    my_plots = [
        {'title': 'Target Correlation', 'desc': 'Distribution of High vs Low risk in the dataset.', 'file': 'target_correlation.png'},
        {'title': 'Correlation Heatmap', 'desc': 'Feature correlation mapping.', 'file': 'correlation_heatmap.png'},
        {'title': 'Algorithm Accuracy', 'desc': 'Accuracy comparison across the trained machine learning models.', 'file': 'model_comparison.png'},
        {'title': 'ROC Curves', 'desc': 'Receiver Operating Characteristic evaluation curves.', 'file': 'roc_curves.png'},
        {'title': 'Feature Importance', 'desc': 'Most impactful clinical features (Random Forest).', 'file': 'feature_importance.png'},
        {'title': 'Confusion Matrix', 'desc': 'True vs. False positive/negative rates.', 'file': 'confusion_matrices.png'}
    ]
    return render_template('index.html', plots=my_plots)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        raw  = body.get('features', {})
        chosen_model = body.get('model', 'Random Forest')

        model_input = []
        for f in feat_names:
            if f in raw:
                model_input.append(float(raw[f]))
            elif f in HIDDEN_DEFAULTS:
                model_input.append(float(HIDDEN_DEFAULTS[f]))
            else:
                model_input.append(0.0)

        arr = np.array([model_input])
        df_in = pd.DataFrame(arr, columns=feat_names)
        df_in = pd.DataFrame(imputer.transform(df_in), columns=feat_names)
        df_in[scaler_cols] = scaler.transform(df_in[scaler_cols])

        model = MODELS.get(chosen_model, MODELS['Random Forest'])
        pred  = int(model.predict(df_in)[0])
        prob  = float(model.predict_proba(df_in)[0][1])

        all_probs = {name: round(float(m.predict_proba(df_in)[0][1])*100, 1) for name, m in MODELS.items()}
        tier_id, tier_label, tier_color, tier_desc = risk_tier(prob)
        flags, tips = clinical_analysis(raw, pred)

        return jsonify({
            'success': True,
            'prediction': pred,
            'probability': round(prob*100, 1),
            'risk_tier': tier_id,
            'risk_label': tier_label,
            'risk_color': tier_color,
            'risk_desc': tier_desc,
            'model_used': chosen_model,
            'all_model_probs': all_probs,
            'flags': flags,
            'tips': tips,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)