from django.db import models

# Create your models here.
class drona(models.Model):
    inputs = models.CharField(max_length=1000)
