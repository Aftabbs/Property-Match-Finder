import pandas as pd
import numpy as np
import re
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    user_data = pd.read_excel(file_path, sheet_name='User Data')
    property_data = pd.read_excel(file_path, sheet_name='Property Data')
    return user_data, property_data

def preprocess_data(user_data, property_data):
    user_data['Budget'] = user_data['Budget'].apply(lambda x: int(re.sub('[^0-9]', '', str(x))))
    property_data['Price'] = property_data['Price'].apply(lambda x: int(re.sub('[^0-9]', '', str(x))))
    property_data = property_data.rename(columns={'Price': 'Budget'})

    scaler = MinMaxScaler()
    num_features = ['Budget', 'Bedrooms', 'Bathrooms']
    
    user_data[num_features] = scaler.fit_transform(user_data[num_features])
    property_data[num_features] = scaler.transform(property_data[num_features])
    
    return user_data, property_data

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') ## Load Pre traind model for sentence embeddings 

def compute_embeddings(text):
    return embedding_model.encode([text])[0]

def calculate_match_score(user, property_):
    numeric_score = 1 - np.linalg.norm(user[['Budget', 'Bedrooms', 'Bathrooms']].values - property_[['Budget', 'Bedrooms', 'Bathrooms']].values)
    user_embedding = compute_embeddings(user['Qualitative Description'])
    property_embedding = compute_embeddings(property_['Qualitative Description'])
    text_similarity = cosine_similarity([user_embedding], [property_embedding])[0][0]
    match_score = 0.5 * numeric_score + 0.5 * text_similarity
    return round(match_score * 100, 2) 

def match_properties(user_id):
    user = user_data[user_data['User ID'] == user_id].iloc[0]
    scores = []
    for _, property_ in property_data.iterrows():
        score = calculate_match_score(user, property_)
        scores.append((property_['Property ID'], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

user_data, property_data = load_data("Case Study 2 Data.xlsx")
user_data, property_data = preprocess_data(user_data, property_data)
scores = match_properties(1)

score_dict = {prop_id: score for prop_id, score in scores}

def get_match_score(property_id):
    if property_id not in score_dict:
        return "Invalid Property ID", "Please select a valid Property ID."

    score = score_dict[property_id]
    
    if score > 50:
        comment = "**Great Match!** This property closely aligns with your preferences."
    elif 30 <= score <= 50:
        comment = "**Moderate Match.** Some preferences align, but there might be trade-offs."
    else:
        comment = "**Low Match.** This property may not meet your requirements."

    return f"**Match Score:** {score}%", comment

property_ids = [prop_id for prop_id, _ in scores]  

with gr.Blocks() as demo:
    gr.Markdown("# Agent MIRA Property Match Finder")
    gr.Markdown("### Select a Property ID to check its match score.")

    property_dropdown = gr.Dropdown(choices=property_ids, label="Select Property ID", interactive=True)
    output_score = gr.Markdown()
    output_comment = gr.Markdown()

    property_dropdown.change(get_match_score, inputs=[property_dropdown], outputs=[output_score, output_comment])

demo.launch()
