from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog
from django.shortcuts import redirect, reverse, render
from django.views import generic
from django.db.models import Q
from django.db.utils import IntegrityError
from django.contrib import messages
from django.utils import timezone

from datetime import timedelta
import pandas as pd


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


def data_upload(request):
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a csv file')
    deck_id = request.POST["deck_id"]

    data = pd.read_csv(csv_file, sep='\t')
    data.dropna(inplace=True, axis=0)
    for i in range(data.shape[0]):
        card =data.iloc[i]
        question_text = card["up"]
        answer_text = card["down"]
        kwargs = {
            'deck_id': deck_id,
            'question_text': question_text,
            'answer_text': answer_text,
            'association_text': ''
        }
        card = Card(**kwargs)
        card.save()
    # except:
    #     messages.error(request, 'This file can\'t export as data')

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
    else:
        is_good = False
        new_experience = 0

    last_remembered = timezone.now()
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
            days = []
            for day in range(7):
                cards_to_learn = 0
                for card in cards:
                    if card.time_to_be_remembered(when=timezone.now() + timedelta(days=day))<0:
                        cards_to_learn += 1
                        card.experience += 1
                        card.last_remembered = timezone.now() + timedelta(days=day)
                days.append(cards_to_learn)
            cards_to_remember[deck.name] = days
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
            return redirect('decks')
        return super(LearnView, self).get(*args, **kwargs)

    def get_queryset(self):
        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))
        card_to_remember = None
        lowest_time_to_remember = 0
        cards_to_remember = 0
        for card in cards:
            if card.time_to_be_remembered() < 0:
                cards_to_remember += 1
            card_time = card.time_to_be_remembered()
            if card_time<lowest_time_to_remember:
                lowest_time_to_remember = card_time
                card_to_remember = card
        return card_to_remember, cards_to_remember


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