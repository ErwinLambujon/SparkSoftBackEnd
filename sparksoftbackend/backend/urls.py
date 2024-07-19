from django.urls import path
from . import views

print("Loading URLs for your_app_name")

urlpatterns = [
    path('upload_files/', views.upload_files, name='upload_files'),
    path('ask_ai/', views.ask_ai, name='ask_ai'),
]
