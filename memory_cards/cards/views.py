from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog
from django.shortcuts import redirect, reverse
from django.views import generic
from django.db.models import Q
from django.db.utils import IntegrityError
from django.contrib import messages


def add_deck_form(request):
    name = request.POST["name"]
    kwargs = {'name': name}
    deck = Deck(**kwargs)
    try:
        deck.save()
    except IntegrityError as err:
        messages.error(request, err)
    return redirect('decks')


class CardsView(generic.ListView):
    template_name = 'cards/cards.html'
    context_object_name = 'deck'

    def get_queryset(self):
        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        return Card.objects.filter(Q(deck=deck))


class IndexView(generic.ListView):
    template_name = 'cards/decks.html'
    context_object_name = 'decks'

    def get_queryset(self):
        order_by = 'name'
        # return Deck.objects.order_by(order_by), UserDeck.objects.filter(Q(user=self.request.user)).order_by(order_by)
        return Deck.objects.order_by(order_by)


def deck_delete(request, deck_name):
    Deck.objects.filter(name=deck_name).delete()
    return redirect('decks')
