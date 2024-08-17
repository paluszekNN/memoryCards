from django.db import models


class Deck(models.Model):
    name = models.CharField(max_length=1000, unique=True)


class Card(models.Model):
    deck = models.ForeignKey(Deck, on_delete=models.CASCADE)
    question_text = models.CharField(max_length=1000, unique=True)
    answer_text = models.CharField(max_length=1000)
    association_text = models.CharField(max_length=1000)
    last_remembered = models.DateTimeField("last remembered", default=None, null=True)
    experience = models.FloatField(default=0)


class CardLog(models.Model):
    card = models.ForeignKey(Card, on_delete=models.CASCADE)
    association_text = models.CharField(max_length=1000)
    time_diff_min = models.FloatField()
    is_good = models.BooleanField()
