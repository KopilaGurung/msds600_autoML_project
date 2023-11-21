import pandas as pd 
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    df = pd.read_csv(filepath, index_col='customerID')
    return df

def make_predictions(df):
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, data=df)
    return predictions


if __name__ == "__main__" :
    df = load_data('churn_data.csv')
    predictions = make_predictions(df)
    print('predcitions:' )
    print(predictions)
    
    