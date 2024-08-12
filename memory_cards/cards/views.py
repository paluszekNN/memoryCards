from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog
from django.shortcuts import redirect
from django.views import generic
from django.db.models import Q


def add_deck(request):
    # if not request.user.is_superuser:
    #     return HttpResponseForbidden('Nope!')
    template = loader.get_template('cards/add_deck.html')
    # print(request.user.last_login)
    return HttpResponse(template.render({}, request))


def add_deck_form(request):
    template = loader.get_template('cards/add_deck.html')
    name = request.POST["name"]
    # print(request.FILES)

    kwargs = {'name': name}
    deck = Deck(**kwargs)
    deck.save()
    return HttpResponse(template.render({}, request))


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