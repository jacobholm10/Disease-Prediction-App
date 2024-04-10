import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np


diabetes_model = pickle.load(open(r'c:\Users\jacob\Desktop\Disease Classification App\diabetes_model.pkl', 'rb'))
cardiovascular_model = pickle.load(open(r'c:\Users\jacob\Desktop\Disease Classification App\cardiovascular_prediction.pkl', 'rb'))
heart_attack_model = pickle.load(open(r'c:\Users\jacob\Desktop\Disease Classification App\heart_model.pkl', 'rb'))




def scale_user_input(user_input):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(user_input).reshape(-1, 1))
    return scaled_data


def validate_form(form_data):
    for value in form_data.values():
        if not value.strip():
            return False
    return True
