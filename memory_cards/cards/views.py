from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog
from django.shortcuts import redirect, reverse
from django.views import generic
from django.db.models import Q
from django.db.utils import IntegrityError
from django.contrib import messages
from django.utils import timezone


def add_deck_form(request):
    name = request.POST["name"]
    kwargs = {'name': name}
    deck = Deck(**kwargs)
    try:
        deck.save()
    except IntegrityError as err:
        messages.error(request, err)
    return redirect('decks')


def add_card_form(request):
    deck_id = request.POST["deck_id"]
    question_text = request.POST["question_text"]
    answer_text = request.POST["answer_text"]
    association_text = request.POST["association_text"]
    kwargs = {
        'deck_id': deck_id,
        'question_text': question_text,
        'answer_text': answer_text,
        'association_text': association_text
              }
    card = Card(**kwargs)
    try:
        card.save()
    except IntegrityError as err:
        messages.error(request, err)
    return redirect(reverse('cards') + '?q=' + deck_id)


def log_card_form(request):
    id_card = request.POST["id_card"]
    card_q = Card.objects.filter(
        Q(id__icontains=id_card)
    )
    card = card_q[0]
    association_text = request.POST["association_text"]
    answer = request.POST['answer']
    time_diff_min = card.last_remember_min()
    if answer == 'yes':
        is_good = True
        new_experience = card.experience + 1
        last_remembered = timezone.now()
    else:
        is_good = False
        new_experience = 0
        last_remembered = card.last_remembered

    kwargs_log = {
        'card_id': card.id,
        'question_text': card.question_text,
        'answer_text': card.answer_text,
        'association_text': association_text,
        'experience': card.experience,
        'time_diff_min': time_diff_min,
        'is_good': is_good
              }
    kwargs_card = {
        'association_text': association_text,
        'experience': new_experience,
        'last_remembered': last_remembered
    }
    card_log = CardLog(**kwargs_log)
    card_log.save()
    deck_id = card.deck.id
    card_q.update(**kwargs_card)
    return redirect(reverse('log_card') + '?q=' + str(deck_id))


class CardsView(generic.ListView):
    template_name = 'cards/cards.html'
    context_object_name = 'cards'

    def get_queryset(self):
        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        return Card.objects.filter(Q(deck=deck))

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['deck_id'] = self.request.GET.get('q')
        return context


class IndexView(generic.ListView):
    template_name = 'cards/decks.html'
    context_object_name = 'decks'

    def get_queryset(self):
        order_by = 'name'
        # return Deck.objects.order_by(order_by), UserDeck.objects.filter(Q(user=self.request.user)).order_by(order_by)
        return Deck.objects.order_by(order_by)

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        cards_to_remember = {}
        for deck in context['decks']:
            cards = Card.objects.filter(Q(deck=deck))
            cards_to_learn = 0
            for card in cards:
                if card.time_to_be_remembered()<0:
                    cards_to_learn += 1
            cards_to_remember[deck.name] = cards_to_learn
        context['cards_to_remember'] = cards_to_remember
        return context


def deck_delete(request, deck_name):
    Deck.objects.filter(name=deck_name).delete()
    return redirect('decks')


def card_delete(request, card_name):
    card = Card.objects.filter(question_text=card_name)
    deck_id = card[0].deck.id
    card.delete()
    return redirect(reverse('cards') + '?q=' + str(deck_id))


class EditView(generic.ListView):
    template_name = 'cards/edit_card.html'
    context_object_name = 'card'

    def get_queryset(self):
        return Card.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]


class LearnView(generic.ListView):
    template_name = 'cards/log_card.html'
    context_object_name = 'learn_card'

    def get(self, *args, **kwargs):
        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))
        cards_to_remember = []
        for card in cards:
            if card.time_to_be_remembered() < 0:
                cards_to_remember.append(card)
        if not cards_to_remember:
            return redirect(reverse('cards') + '?q=' + str(self.request.GET.get('q')))
        return super(LearnView, self).get(*args, **kwargs)

    def get_queryset(self):
        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))
        card_to_remember = None
        lowest_time_to_remember = 0
        for card in cards:
            card_time = card.time_to_be_remembered()
            if card_time<lowest_time_to_remember:
                lowest_time_to_remember = card_time
                card_to_remember = card
        return card_to_remember


def edit_card_form(request):
    id_card = request.POST["id_card"]
    question_text = request.POST["question_text"]
    answer_text = request.POST["answer_text"]
    association_text = request.POST["association_text"]
    kwargs = {
        'question_text': question_text,
        'answer_text': answer_text,
        'association_text': association_text,
              }

    card = Card.objects.filter(
        Q(id__icontains=id_card)
    )
    deck_id = card[0].deck.id
    card.update(**kwargs)
    return redirect(reverse('cards') + '?q=' + str(deck_id))