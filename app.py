from flask import Flask, render_template, request, jsonify
import torch
from sentence_transformers import util
import pickle
from sentence_transformers import SentenceTransformer
import mysql.connector

app = Flask(__name__)

db_config = {
    'host' : 'localhost',
    'user' : 'root',
    'password' : 'Patya2211*',
    'database' : 'farmer_services'
}

# Define available services as example
# services = {
#     'Loan': 'Financial support to improve farming operations and address water shortage.',
#     'Training': 'Technical training on modern agricultural techniques to improve yield.',
#     'Subsidy': 'Subsidized resources to address pest control and other challenges.',
#     'Market Access': 'Support for better access to agricultural markets.',
#     'Soil Testing': 'Analysis of soil health and recommendations for improvement.'
# }


# Load necessary data
embeddings = pickle.load(open('service_embeddings.pkl', 'rb'))
if isinstance(embeddings, dict):
    embeddings = torch.tensor(list(embeddings.values()))
services = pickle.load(open('services.pkl', 'rb'))
rec_model = pickle.load(open('rec_model.pkl', 'rb'))

# Load sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute service embeddings
# service_descriptions = list(services.values())
# service_embeddings = model.encode(service_descriptions, convert_to_tensor=True)

# Recommendation function
def recommendation(farmer_issues):
    """Recommends services based on the farmer's issues using cosine similarity."""

    cosine_scores = util.cos_sim(rec_model.encode(farmer_issues, convert_to_tensor=True), embeddings)

    top_results = torch.topk(cosine_scores, k=3) # get top 3 recommendation

    indices = top_results.indices[0].tolist() # extract the indices of top result
    
    recommended_services = [
        list(services.keys())[idx] for idx in indices
        ]
    return recommended_services



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/recommendation', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':

        farmer_issues = request.form.get('farmer_issues', '').strip()
        soil_type = request.form.get('soil_type', '').strip()
        irrigation_practices = request.form.get('irrigation_practices', '').strip()
        farm_type = request.form.get('farm_type', '').strip()
        crop_preferences = request.form.get('crop_preferences', '').strip()
        farm_size = request.form.get('farm_size', '').strip()
        farming_experience = request.form.get('farming_experience', '').strip()
        education_level = request.form.get('education_level', '').strip()
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO farmer_profile (
                farmer_issues, soil_type, irrigation_practices, farm_type,
                crop_preferences, farm_size, farming_experience, education_level
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ''', (farmer_issues, soil_type, irrigation_practices, farm_type, crop_preferences, farm_size, farming_experience, education_level))
        conn.commit()
        cursor.close()
        conn.close()
        print("data added successfully")
        if farmer_issues:
            recommendations = recommendation(farmer_issues)
            return render_template('result.html', recommendations=recommendations, services=services)
        
        else:
            return render_template('index.html', error="Please enter farmer issues.")
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
