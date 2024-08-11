from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog


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