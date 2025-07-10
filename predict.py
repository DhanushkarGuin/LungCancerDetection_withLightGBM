import pickle
import pandas as pd

pipeline = pickle.load(open('pipeline.pkl','rb'))

columns = ['age', 'gender', 'cancer_stage', 'family_history', 'smoking_status', 'bmi', 'cholesterol_level', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type', 'diagnosis_year', 'diagnosis_month', 'end_treatment_year', 'end_treatment_month']

test_input = pd.DataFrame([[21, 'Male', 'Stage I', 'Yes', 'Never Smoked', 44, 183, 0, 0, 0, 0, 'Surgery', 2022, 10, 2024, 10]], columns = columns)

prediction = pipeline.predict(test_input)
print('Survived' if prediction[0] == 0 else 'Not survived')