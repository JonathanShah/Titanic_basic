from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
# Create your views here.

     
import sklearn
from sklearn import preprocessing
import joblib


reloadModel=joblib.load('./models/titanic_model.pkl')

def index(request):
    context = {'a':'hello'}
    return render(request,'index.html',context)
    return HttpResponse('a')

def predictMPG(request):
    print(request)
    if request.method == 'POST':
        temp={}
        temp['Pclass']=request.POST.get('Pclass')
        temp['Sex_female']=request.POST.get('Sex_female')
        temp['Sex_male']=request.POST.get('Sex_male')
    testData=pd.DataFrame({'x':temp}).transpose()
    scoreval=reloadModel.predict(testData)[0]

    context={'scoreval':scoreval}
    return render(request,'index.html',context)
    

    context={'a':'hello new'}
    return render(request,'index.html',context)
