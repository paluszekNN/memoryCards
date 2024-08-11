from django.contrib import admin
from .models import Card, Deck, CardLog

admin.site.register(Card)
admin.site.register(CardLog)
admin.site.register(Deck)