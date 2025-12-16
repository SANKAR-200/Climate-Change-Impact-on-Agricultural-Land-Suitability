from django.shortcuts import render
from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Create your views here.
def home(request):
    return render(request,'home.html')


def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
import joblib
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from catboost import CatBoostClassifier
import pandas as pd
import joblib


def prediction(request):
    import pandas as pd
    import joblib
    from django.shortcuts import render

    # Load the trained model and encoders
    clf = joblib.load('model/RandomForestClassifier.pkl')
    encoders = joblib.load('model/label_encoders.pkl')
    target_encoder = joblib.load('model/target_encoder.pkl')

    feature_labels = {
        "Soil_Quality_Index": "Soil Quality Index",
        "Annual_Rainfall_mm": "Annual Rainfall (mm)",
        "Temperature_C": "Temperature (°C)",
        "Irrigation_Level": "Irrigation Level",
        "CO2_Emission_Scenario": "CO2 Emission Scenario",
        "Water_Availability_Index": "Water Availability Index",
        "Fertilizer_Access": "Fertilizer Access",
        "Land_Degradation_Risk": "Land Degradation Risk"
    }

    if request.method == "POST":
        input_data = {}
        for feature in feature_labels:
            value = request.POST.get(feature)
            try:
                input_data[feature] = float(value)
            except ValueError:
                input_data[feature] = value  # Keep categorical as string

        df = pd.DataFrame([input_data])

        # Apply saved encoders to categorical features
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        prediction_result = clf.predict(df)[0]
        outcome = target_encoder.inverse_transform([prediction_result])[0]

        return render(request, 'outcome.html', {'outcome': outcome})

    return render(request, 'prediction.html', {"feature_labels": feature_labels})




from django.core.files.storage import default_storage
le=LabelEncoder()
dataloaded=False
global X_train,X_test,y_train,y_test
global df
def Upload_data(request):
    load=True
    global df,dataloaded
    global X_train,X_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        df=pd.read_csv(default_storage.path(file_path))
        le=LabelEncoder()
        categorical_cols = ['Irrigation_Level', 'CO2_Emission_Scenario', 'Fertilizer_Access', 'Land_Degradation_Risk']
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        

        sns.set(style="darkgrid")  # Set the style of the plot
        plt.figure(figsize=(8, 6))  # Set the figure size
        ax = sns.countplot(x='Land_Suitability', data=df)
        plt.title("Count Plot")  # Add a title to the plot
        plt.xlabel("Categories")  # Add label to x-axis
        plt.ylabel("Count")  # Add label to y-axis
        # Annotate each bar with its count value
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.xticks(rotation=90)
        plt.show()
        x=df.iloc[:,1:-1]
        y=df.iloc[:,-1]
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        joblib.dump(encoders, 'model/label_encoders.pkl')
        joblib.dump(target_encoder, 'model/target_encoder.pkl')
        X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=42)
        default_storage.delete(file_path)
        outdata=df.head(100)
        dataloaded=True
        return render(request,'train.html',{'temp':outdata.to_html()})
    return render(request,'train.html',{'upload':load})
labels=['Unsuitable', 'Moderate', 'Highly Suitable']
#defining global variables to store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
def calculateMetrics(algorithm, testY,predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    
def RFC(request):
    if dataloaded == False:
        return redirect('upload')
    
    model_path = 'model/RandomForestClassifier.pkl'
    os.makedirs('model', exist_ok=True)

    if os.path.exists(model_path):
        # Load the trained model from the file
        rf = joblib.load(model_path)
        print("RandomForestClassifier model loaded successfully.")
        predict = rf.predict(X_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(rf, model_path)
        print("RandomForestClassifier model saved successfully.")
        predict = rf.predict(X_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)

    return render(request, 'train.html',
                  {'algorithm': 'Random Forest Classifier',
                   'accuracy': accuracy[-1],
                   'precision': precision[-1],
                   'recall': recall[-1],
                   'fscore': fscore[-1]})

def KNN(request):
    if dataloaded == False:
        return redirect('upload')
    
    # Flatten input if it's 3D
    if len(X_train.shape) == 3:
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_reshaped = X_train
        X_test_reshaped = X_test

    model_path = 'model/KNNClassifier.pkl'
    os.makedirs('model', exist_ok=True)

    if os.path.exists(model_path):
        knn = joblib.load(model_path)
        print("KNN model loaded successfully.")
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_reshaped, y_train)
        joblib.dump(knn, model_path)
        print("KNN model saved successfully.")
    
    predict = knn.predict(X_test_reshaped)
    calculateMetrics("KNN", predict, y_test)

    return render(request, 'train.html',
                  {'algorithm': 'K-Nearest Neighbors (KNN)',
                   'accuracy': accuracy[-1],
                   'precision': precision[-1],
                   'recall': recall[-1],
                   'fscore': fscore[-1]})

def SVM(request):
    if dataloaded == False:
        return redirect('upload')

    # Ensure the model directory exists
    model_path = 'model/SVM_model.pkl'
    os.makedirs('model', exist_ok=True)

    if os.path.exists(model_path):
        # Load the trained SVM model
        svm = joblib.load(model_path)
        print("SVM model loaded successfully.")
    else:
        # Train the SVM model
        svm = SVC(kernel='rbf', probability=True)  # Can use 'linear', 'poly', etc.
        svm.fit(X_train, y_train)
        joblib.dump(svm, model_path)
        print("SVM model saved successfully.")
    
    predict = svm.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

    return render(request, 'train.html',
                  {'algorithm': 'Support Vector Machine (SVM)',
                   'accuracy': accuracy[-1],
                   'precision': precision[-1],
                   'recall': recall[-1],
                   'fscore': fscore[-1]})
