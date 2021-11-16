from django.urls import path
from .views import dronaList
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    # path('', views.home, name='home'),
    path('', views.dronaList.final, name='final')
]

urlpatterns += staticfiles_urlpatterns()