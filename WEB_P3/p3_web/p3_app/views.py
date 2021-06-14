from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import Input

def upload(request):
    return render(request,'index.html')

def upload_create(request):
    form=Input()
    try:
        form.image=request.FILES['image']
    except: #이미지가 없을 때
        pass
    form.save()

    # return redirect('result/') 
    # return HttpResponse('result')
    return render(request,'result.html',{'form':form})