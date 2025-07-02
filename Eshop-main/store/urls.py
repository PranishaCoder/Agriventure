from django.contrib import admin
from django.urls import path
from .views.home import Index , store
from .views.signup import Signup
from .views.login import Login , logout
from .views.cart import Cart
from .views.checkout import CheckOut
from .views.orders import OrderView
from .middlewares.auth import  auth_middleware
from django.urls import path, include
# from .views.upload import Upload
from prescription import views

admin.site.site_header="Admin Login Portal"
admin.site.site_title=" Delivery Admin Portal"
admin.site.index_title=" Delivery Admin Database "

urlpatterns = [
    path('', Index.as_view(), name='homepage'),
    path('store', store , name='store'),

    path('signup', Signup.as_view(), name='signup'),
    path('login', Login.as_view(), name='login'),
    path('logout', logout , name='logout'),
    path('cart', auth_middleware(Cart.as_view()) , name='cart'),
    path('check-out', CheckOut.as_view() , name='checkout'),
    path('orders', auth_middleware(OrderView.as_view()), name='orders'),

    path("upload", views.prescrip, name='upload'),
    path('index1', views.index1, name='index1'),
    path('predict_weather', views.predict_weather, name='predict_weather'),
    path('index2', views.index2, name='index2'),
    path('crop_prediction', views.crop_prediction, name='crop_prediction'),
    path('crop_type/', views.crop_type, name='crop_type'),
    path('predict1', views.predict1, name='predict1'),
    path('plant-disease/', views.plant_disease_view, name='plant_disease'),

    
]
