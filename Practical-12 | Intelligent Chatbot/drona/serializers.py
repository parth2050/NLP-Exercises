from rest_framework import serializers
from .models import drona

class dronaSerializer(serializers.ModelSerializer):

    class Meta:
        model = drona
        fields = '__all__'



