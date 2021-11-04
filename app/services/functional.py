from services.predict import MachineLearningModelHandler as model
import joblib


def predict(image):
    return model.predict(image)