from django.db import models


class Card(models.Model):
    question_text = models.CharField(max_length=1000)
    answer_text = models.CharField(max_length=1000)
    association_text = models.CharField(max_length=1000)
    last_remembered = models.DateTimeField("last remembered", default=None, null=True)
    experience = models.FloatField(default=0)
