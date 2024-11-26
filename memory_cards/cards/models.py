from django.utils import timezone

from django.db import models


class Deck(models.Model):
    name = models.CharField(max_length=1000, unique=True)


class Card(models.Model):
    deck = models.ForeignKey(Deck, on_delete=models.CASCADE)
    question_text = models.CharField(max_length=1000, unique=True)
    answer_text = models.CharField(max_length=1000)
    association_text = models.CharField(max_length=1000)
    last_remembered = models.DateTimeField("last remembered", default=timezone.now())
    experience = models.FloatField(default=0)

    def last_remember_min(self, when=None):
        if not when:
            when = timezone.now()
        difference = when - self.last_remembered
        # print(when)
        # print(self.last_remembered)
        # print(difference)
        # print(difference.total_seconds() / 60)
        return difference.total_seconds() / 60

    def time_to_be_remembered(self, when=None):
        if not when:
            when = timezone.now()
        if self.experience == 0:
            return -1
        else:
            return self.experience**2 * 1440 - self.last_remember_min(when)

    def split_name(self):
        return self.question_text.split(' ')


class CardLog(models.Model):
    card = models.ForeignKey(Card, on_delete=models.CASCADE)
    question_text = models.CharField(max_length=1000)
    answer_text = models.CharField(max_length=1000)
    association_text = models.CharField(max_length=1000)
    experience = models.FloatField(default=0)
    time_diff_min = models.FloatField()
    is_good = models.BooleanField()
