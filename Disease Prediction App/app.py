from flask import Flask, render_template, request, redirect
from models import scale_user_input, diabetes_model, heart_attack_model, cardiovascular_model, validate_form




app = Flask(__name__)



@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/select_model", methods=['POST'])
def select_model():
    selected_model = request.form['model']
    if selected_model == 'cardiovascular':
        return redirect('/cardiovascular')
    elif selected_model == 'heart':
        return redirect('/heart')
    elif selected_model == 'diabetes':
        return redirect('/diabetes')


@app.route("/cardiovascular")
def cardiovascular_input():
    return render_template('cardiovascular_input.html')


@app.route("/heart")
def heart_input():
    return render_template('heart_input.html')


@app.route("/diabetes")
def diabetes_input():
    return render_template('diabetes_input.html')




@app.route("/cardiovascular/predict", methods=['POST'])
def predict_cardiovascular_model():
    if not validate_form(request.form):
        return render_template('cardiovascular_input.html', error_message="Please fill in all fields")
    
    float_features = [float(x) for x in request.form.values()]
    scaled_features = scale_user_input(float_features)
    scaled_features_reshaped = scaled_features.reshape(1, -1)
    prediction = cardiovascular_model.predict(scaled_features_reshaped)
    if prediction[0] == 0:
        prediction_message = "You do not have cardiovascular disease"
    else:
        prediction_message = "You have a high chance of having cardiovascular disease"

    return render_template('prediction_result.html', prediction_message=prediction_message)




@app.route('/heart/predict', methods=['POST'])
def predict_heart_model():
    if not validate_form(request.form):
        return render_template('heart_input.html', error_message="Please fill in all fields")
    
    float_features = [float(x) for x in request.form.values()]
    scaled_features = scale_user_input(float_features)
    scaled_features_reshaped = scaled_features.reshape(1, -1)
    prediction = heart_attack_model.predict(scaled_features_reshaped)
    if prediction[0] == 0:
        prediction_message = "You do not have a heart disease"
    else:
        prediction_message = "You have a high chance of having a heart disease"

    return render_template('prediction_result.html', prediction_message=prediction_message)



@app.route('/diabetes/predict', methods=['POST'])
def predict_diabetes_model():
    if not validate_form(request.form):
        return render_template('diabetes_input.html', error_message="Please fill in all fields")
    
    float_features = [float(x) for x in request.form.values()]
    scaled_features = scale_user_input(float_features)
    scaled_features_reshaped = scaled_features.reshape(1, -1)
    prediction = diabetes_model.predict(scaled_features_reshaped)
    if prediction[0] == 0:
        prediction_message = "You do not have diabetes"
    else:
        prediction_message = "You have a high chance of having diabetes"

    return render_template('prediction_result.html', prediction_message=prediction_message)





if __name__ == '__main__':
    app.run(debug=True)

    


