import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

#model = joblib.load('model/loan_pipe.pkl')
file_name = 'model/sberavto2.pkl'
with open(file_name, 'rb') as file:
    model = dill.load(file)

#
class Form(BaseModel):
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    month: int
    day: int
    hour: int

class FormOriginal(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str

class Prediction(BaseModel):
    pred: int


@app.post('/predict', response_model=Prediction)
def predict(form: FormOriginal):
    def delete(df):
        df = df.drop(['session_id', 'device_model', 'client_id'], axis=1)

    def date_time(df):
        df['month'] = df.visit_date.apply(lambda x: int(x.split('-')[1]))
        df['day'] = df.visit_date.apply(lambda x: int(x.split('-')[2]))
        df['hour'] = df.visit_time.apply(lambda x: int(x.split(':')[0]))

    def delete2(df):
        df = df.drop(['visit_date', 'visit_time'], axis=1)

    df = pd.DataFrame.from_dict([form.dict()])
    delete(df)
    date_time(df)
    delete2(df)
    y = model['model'].predict(df)

    return {
        "pred": y[0]
    }


@app.get('/status')
def status():
    return "I'm OK!"


@app.get('/version')
def version():
    return model['metadata']

def main():
    def delete(df):
        df = df.drop(['session_id', 'device_model', 'client_id'], axis=1)


    def date_time(df):
        df['month'] = df.visit_date.apply(lambda x: int(x.split('-')[1]))
        df['day'] = df.visit_date.apply(lambda x: int(x.split('-')[2]))
        df['hour'] = df.visit_time.apply(lambda x: int(x.split(':')[0]))

    def delete2(df):
        df = df.drop(['visit_date', 'visit_time'], axis=1)

    with open('model/data/2.json') as json_file:
        test = json.load(json_file)
        #q = dict(test)
    df = pd.DataFrame.from_dict([test])
    delete(df)
    date_time(df)
    delete2(df)
    y = model['model'].predict(df)

    print(y[0])


if __name__ == '__main__':
    main()

