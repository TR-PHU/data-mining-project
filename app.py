from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

def load_data(filename):
    data = pd.read_csv(filename)
    selected_columns = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan']
    data = data[selected_columns]
    return data

def preprocess_data(data):    
    categorical_cols = ['job', 'marital', 'education', 'housing', 'loan']
    
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    scaler = StandardScaler()
    data[['age', 'balance']] = scaler.fit_transform(data[['age', 'balance']])
    
    return data.values

def train_gmm(X, num_components):
    gmm = GaussianMixture(n_components=num_components, random_state=0)
    gmm.fit(X)
    return gmm

def predict_segmentation(gmm, new_customer):
    return gmm.predict(new_customer.reshape(1, -1))

def add_and_preprocess_data(df, new_data):
    new_df = pd.DataFrame([new_data])

    df = df.append(new_df, ignore_index=True)

    preprocessed_df = preprocess_data(df)

    return preprocessed_df

filename = 'bank.csv'
data = load_data(filename)

preprocessed_data = preprocess_data(data)
num_components = 3
gmm = train_gmm(preprocessed_data, num_components)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # Chuẩn bị dữ liệu để tiền xử lý
        customer_data = {
            'age': int(request.form['age']),
            'job': request.form['job'],
            'marital': request.form['marital'],
            'education': request.form['education'],
            'balance': int(request.form['balance']),
            'housing': request.form['housing'],
            'loan': request.form['loan']
        }
        data = load_data(filename)
        data = add_and_preprocess_data(data, customer_data)

        # Dự đoán phân đoạn cho khách hàng mới
        new_customer = data[-1]
        segment = predict_segmentation(gmm, new_customer)
        
        return render_template('index.html', segment=segment[0]+1)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
