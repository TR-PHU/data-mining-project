from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize, OrdinalEncoder
from sklearn.decomposition import PCA

app = Flask(__name__)

class GMM_model:
    __model = None
    __encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    __scaler = StandardScaler() 
    __pca =  PCA(2) 
    
    def __init__(self):
        df = pd.read_csv("./bank.csv")
        df = df.drop('contact', axis=1)
        df = df.dropna()

        X_principal = self.__init_preprocessing(df)
        
        # Fitting model 
        self.__model = GaussianMixture(n_components=4).fit(X_principal)
        
    def predict(self, df):
        # Encoding df
        encoded_df = self.__encoder.transform(df)
        encoded_df = pd.DataFrame(encoded_df)
        encoded_df.columns = df.columns.tolist()      
        
        print("Encoded df: ", encoded_df)
        # Scaling
        scaled_df = self.__scaler.transform(encoded_df) 
        
        # Normalizing 
        normalized_df = normalize(scaled_df) 
        normalized_df = pd.DataFrame(normalized_df) 
        
        # Reducing the dimensions of the data 
        X_principal = self.__pca.transform(normalized_df) 
        X_principal = pd.DataFrame(X_principal) 
        
        return self.__model.predict(X_principal) 
    
    def __init_preprocessing(self, df):
        # Encoding df
        self.__encoder.fit(df)
        encoded_df = self.__encoder.transform(df)
        encoded_df = pd.DataFrame(encoded_df)
        encoded_df.columns = df.columns.tolist()

        # Scaling
        self.__scaler.fit(encoded_df)
        scaled_df = self.__scaler.transform(encoded_df) 
        
        # Normalizing 
        normalized_df = normalize(scaled_df) 
        normalized_df = pd.DataFrame(normalized_df) 
        
        # Reducing the dimensions of the data 
        self.__pca.fit(normalized_df)
        X_principal = self.__pca.transform(normalized_df) 
        X_principal = pd.DataFrame(X_principal) 
        
        return X_principal

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        age = int(request.form['age'])
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        default = request.form['default']
        balance = int(request.form['balance'])
        housing = request.form['housing']
        loan = request.form['loan']
        day = int(request.form['day'])
        month = request.form['month']
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])
        pdays = int(request.form['pdays'])
        previous = int(request.form['previous'])
        poutcome = request.form['poutcome']
        deposit = request.form['deposit']

        new_customer = pd.DataFrame([[age, job, marital, education, default, balance, housing, loan,
                        day, month, duration, campaign, pdays, previous, poutcome, deposit]])
        new_customer.columns = pd.read_csv('./bank.csv').drop(['contact'], axis=1).columns.tolist()

        model = GMM_model()
        segment = model.predict(new_customer)
        
        return render_template('index.html', segment=segment[0]+1)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
