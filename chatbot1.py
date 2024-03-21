from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
data = [
    ("headache fever cough", "Common Cold"),
    ("fever chills fatigue", "Flu"),
    ("rash itching redness", "Allergy"),
      ("rash itching stomach pain", "Drug reaction"), 
      ("fatigue restlessness obesity", "Diabetese"),
     ("Cough Fatigue Breathlessness", "Asthama"), 
     ("Headache dizziness Chest Pain", "Hypertension"),
       ("rash itching redness", "Allergy"),
      ("High Fever Yellow skin Vomiting", "Jaundice"),
        ("High fever Chills Nausea muscle pain ", "Malaria"),
          ("Mild fever Skinrash Itching", "Chicken-Pox"),
      ("Highfever Red Spots Nausia", "Dengue"),
      ("Diarrhoea High fever Vomiting", "Typhoid"),
      ("Breathlessness Cough Sweating", "Tuberculosis"),
      (" joint pain vomiting yellowish skin dark urine nausea","hepatitis A"),
      ("continuous sneezing shivering watering from eyes","Allergy"),
      ("stomach pain ulcers on tongue vomiting cough chest pain","GERD"),
      (" vomiting loss of appetite abdominal pain passage of gases","Peptic ulcer diseae"),
      (" muscle wasting patches in throat high fever","AIDS"),
      ("Gastroenteritis vomiting sunken eyes dehydration diarrhoea vomiting sunken eyes dehydration diarrhoea sunken eyes, dehydration, diarrhoea","Gastroenteritis"),


    
]

symptoms, labels = zip(*data)
model = make_pipeline(CountVectorizer(), DecisionTreeClassifier())
model.fit(symptoms, labels)
def predict_disease(symptoms_input):
    predicted_label = model.predict([symptoms_input])
    return predicted_label[0]
user_input = input("Hello!! I am your HealthBuddy. Do you have any symptoms related to any disease?:Y/N ")
if(user_input=='Y'):

    user_input = input("Enter 3 or 4 symptoms separated by spaces: ")
    predicted_disease = predict_disease(user_input)
    print(f"The predicted disease based on symptoms is: {predicted_disease}")
else:
    print("Thank You. I wish you well!!")