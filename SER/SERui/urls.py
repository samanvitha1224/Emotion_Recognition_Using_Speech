from django.urls import path
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns = [
    path('',views.open,name='home-page' ),
    path('predict',views.predict,name='prediction-page')
]
urlpatterns+=staticfiles_urlpatterns()