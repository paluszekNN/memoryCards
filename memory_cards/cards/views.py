from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog
from django.shortcuts import redirect
from django.views import generic
from django.db.models import Q


def add_deck_form(request):
    name = request.POST["name"]
    kwargs = {'name': name}
    deck = Deck(**kwargs)
    deck.save()
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
