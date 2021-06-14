from django.db import models

class Input(models.Model):
    image = models.ImageField(upload_to='images/')
