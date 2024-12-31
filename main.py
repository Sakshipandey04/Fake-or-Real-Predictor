from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from fastapi.responses import JSONResponse

# Load the trained model and vectorizers
model = joblib.load('classification_model.pkl')
tfidf_title = joblib.load('tfidf_title.pkl')  
tfidf_text = joblib.load('tfidf_text.pkl')    


app = FastAPI()


class InputData(BaseModel):
    title: str
    text: str  


def extract_features(title, text):

    title_tfidf = tfidf_title.transform([title])
    text_tfidf = tfidf_text.transform([text])


    title_length = len(title.split())
    text_length = len(text.split())
    sentiment_polarity = TextBlob(text).sentiment.polarity
    sentiment_subjectivity = TextBlob(text).sentiment.subjectivity
    keyword_density = sum(text.count(keyword) for keyword in ['important', 'keyword1', 'keyword2']) / len(text.split())

 
    features = np.hstack([title_tfidf.toarray(), text_tfidf.toarray(),
    np.array([[title_length, text_length, keyword_density, sentiment_polarity, sentiment_subjectivity]])])

    return features

@app.post("/predict")
def predict(input_data: InputData):
    print(f"Received input: Title: {input_data.title}, Text: {input_data.text}")
  
    features = extract_features(input_data.title, input_data.text)
    
  
    prediction = model.predict(features)
    
   
    label = int(prediction[0])
    result = "real" if label == 1 else "fake"
    return JSONResponse(content= {"prediction": result, "label": label})
    