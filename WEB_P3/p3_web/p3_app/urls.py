from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('',views.upload,name="upload"),
    path('result/',views.upload_create,name="upload_create"),
    path('by_class/',views.detect_by_class,name="detect_by_class"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)