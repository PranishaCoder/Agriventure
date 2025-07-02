from django.db import models

# Create your models here.
import datetime


class Prescription(models.Model):

    fullname = models.CharField(max_length=260)
    contactnum = models.CharField(max_length=15,null=True)
    image= models.ImageField(upload_to='uploads/products/')
    date = models.DateField (default=datetime.datetime.today)

    def __str__(self):
        return self.fullname


