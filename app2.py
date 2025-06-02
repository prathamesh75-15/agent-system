from flask import Flask, render_template, request, jsonify, redirect
import torch
from sentence_transformers import util
import pickle
import tensorflow as tf
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


service_images = {
    'Loan': '/static/loan.jpeg',
    'Training': '/static/traning.jpg',
    'Subsidy': '/static/subsidy.jpg',
    'Market Access': '/static/market access.jpg',
    'Soil Testing': '/static/soil testing.jpg',
    'Crop Selection Advisory': '/static/crop selection adivisory.jpg',
    'Weather Alerts': '/static/weather alert.jpg',
    'Irrigation Plans': '/static/irrigations.jpg',
    'Organic Farming Support': '/static/organic farming support.jpg',
    'Precision Farming': '/static/precision farming.jpg',
    'Crop-Specific Training': '/static/organic farming vegitable.jpg',
    'Wheat Monitoring': '/static/wheat monitoring.jpg',
    'Corn Disease Detection': '/static/corn diseases detection.jpg',
    'Rice Water Management': '/static/rice water management.jpg',
    'Vegetable Organic Farming': '/static/vegitable organic farming.jpg',
    'Water Analysis Facility': '/static/Water Analysis Facility.jpg',
    'Tractor Booking Facility':'/static/Tractor Booking Facility.jpg',
    'Seed Selection Advisory':'/static/Seed Selection Advisory.jpg',
    'Fertilizer Recommendation': '/static/Fertilizer Recommendation.jpg',
    'Pest and Disease Control': '/static/Pest and Disease Control.jpg',
    'Weather Forecasting': '/static/Weather Forecasting.jpg',
    'Government Scheme Assistance':'/static/Government Scheme.webp',
    'Rental Equipment Facility': '/static/Rental Equipment Facility.jpg',
    'Insurance Service':'/static/Insurance Service.jpg'
      
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
embeddings = pickle.load(open('service_embeddings_new.pkl', 'rb'))
if isinstance(embeddings, dict):
    embeddings = torch.tensor(list(embeddings.values()))
services = pickle.load(open('services_new.pkl', 'rb'))
rec_model = pickle.load(open('rec_model.pkl', 'rb'))
vectorizer = pickle.load(open('C:\\Users\\DELL\\small recommendation system\\vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('C:\\Users\\DELL\\small recommendation system\\label_encoder.pkl', 'rb'))

# loading of model neural network 
# query_type_model = tf.keras.models.load_model('C:\\Users\\DELL\\small recommendation system\\neural_network_model_for_QueryType.h5')  # Load your query type model
query_type_model = tf.keras.models.load_model("optimized_model.h5") 
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
def services_ranking(scores_list):
    # Get all services with their similarity scores
    all_services_with_scores = [
        (list(services.keys())[idx], scores_list[idx]*100) for idx in range(len(scores_list))
    ]
    all_services_with_scores.sort(key=lambda x: x[1], reverse=True)
    # print(all_services_with_scores)

# Recommendation function
def recommendation(farmer_issues,query_type):
    
    cosine_scores = util.cos_sim(rec_model.encode(farmer_issues, convert_to_tensor=True), embeddings)
    scores_list = cosine_scores[0].tolist()
    services_ranking(scores_list)

    top_results = torch.topk(cosine_scores, k=3) # get top 3 recommendation

    indices = top_results.indices[0].tolist() # extract the indices of top result
    
    recommended_services = [
        list(services.keys())[idx] for idx in indices
        ]
    print(recommended_services)
    print(query_type)

    #services releated to querytype{
    # releated_service_to_queryType =[]
    # for service, description in services.items():
    #     description_embedding = rec_model.encode(description, convert_to_tensor =True)
    #     query_embbeding = rec_model.encode(query_type, convert_to_tensor =True)
    #     similarity_score = util.cos_sim(query_embbeding ,description_embedding )
    #     if similarity_score > 0.3:  # Threshold for similarity
    #         releated_service_to_queryType.append(service)
    #}

    similarity_score= util.cos_sim(rec_model.encode(query_type, convert_to_tensor=True), embeddings)
    top_results = torch.topk(similarity_score, k=3)
    indices1 = top_results.indices[0].tolist()
    releated_service_to_queryType=[
        list(services.keys())[idx1] for idx1 in indices1
    ]

    # print(releated_service_to_queryType)

    all_recommendations = list(set(recommended_services + releated_service_to_queryType))
    print(all_recommendations)

    return all_recommendations


#function for query_type prediction 
def predict_query_type(user_input):

    user_input_cleaned = user_input.lower().replace('[^\w\s]', '')
    user_input_vectorized = vectorizer.transform([user_input_cleaned]).toarray()
    # user_input_vectorized = tf.sparse.reorder(user_input_vectorized)
    query_type = query_type_model.predict(user_input_vectorized)
    predicted_index = np.argmax(query_type, axis=1)

    # Decode the predicted index to get the query type
    predicted_query_type = label_encoder.inverse_transform(predicted_index)

    print(predicted_query_type)
    return predicted_query_type[0].strip()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # query_cleaned = user_input.lower().replace('[^\w\s]', '')  # Normalize text

    # query_vectorized = vectorizer.transform([query_cleaned]).toarray()
    # query_vectorized = query_vectorized.astype(np.float32)
    # interpreter.set_tensor(input_details[0]['index'], query_vectorized)
    # interpreter.invoke()
    # prediction = interpreter.get_tensor(output_details[0]['index'])
    # predicted_index = np.argmax(prediction, axis=1)
    # print(predicted_index)
    # predicted_query_type = label_encoder.inverse_transform(predicted_index)

    # return predicted_query_type[0]
    


@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/tryservice')
def tryservice():
    return render_template('index.html')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':

        farmer_issues = request.form.get('farmer_issues', '').strip()

        if farmer_issues:
            query_type = predict_query_type(farmer_issues)
            recommendations = recommendation(farmer_issues,query_type)
            return render_template('result2.html', recommendations=recommendations, services=services,service_images=service_images)
        
        else:
            return render_template('index.html', error="Please enter farmer issues.")
        
    return render_template('index.html')


@app.route('/api/services')
def get_services():
    # return jsonify([{"name": name, "description": desc} for name, desc in services.items()])
    return jsonify([
        {
            "name": name,
            "description": desc,
            "image": service_images.get(name, "/static/images/default.jpg")  # Default image if not found
        } 
        for name, desc in services.items()
    ])

@app.route('/services')
def services_page():
    return render_template('services.html')

@app.route('/services/<service_name>')
def service_page(service_name):
    formatted_service = service_name.replace("_", " ")  # Convert URL format to match dictionary keys

    if formatted_service == "Rental Equipment Facility":
        return redirect("http://127.0.0.1:5001/")
    # if formatted_service == "Insurance Service":
    #     return render_template(f"services/insurance_services/insurance.html")
    if formatted_service in services:
        return render_template(f"services/{service_name}.html")
    else:
        return "<h2>Service not found</h2>", 404

# @app.route('/comprehensive_form')
# def index2():
#     return render_template("services/comprehensive_form.html")
# @app.route('/basic_crop_form')
# def index3():
#     return render_template("services/basic_crop_form.html")
# @app.route('/pmfby_form')
# def index4():
#     return render_template("services/pmfby_form.html")
# @app.route('/weather_form')
# def index5():
#     return render_template("services/weather_form.html")
# @app.route('/livestock_form')
# def index6():
#     return render_template("services/livestock_form.html")
@app.route('/<form_name>_form')
def render_form(form_name):
    form_template = f"services/{form_name}_form.html"
    try:
        return render_template(form_template)
    except:
        return "Form not found", 404

if __name__ == '__main__':
    app.run(debug=True)
