import time

from django.http import HttpResponse
from django.template import loader
from .models import Deck, Card, CardLog
from django.shortcuts import redirect, reverse, render
from django.views import generic
from django.db.models import Q
from django.db.utils import IntegrityError
from django.contrib import messages
from django.utils import timezone
from sklearn.pipeline import make_pipeline
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.linear_model import (LogisticRegression, Lasso, RidgeClassifier, MultiTaskLasso, LassoLars, LassoLarsIC, PassiveAggressiveClassifier, SGDClassifier, Perceptron, ElasticNet,
                                  OrthogonalMatchingPursuit,BayesianRidge,MultiTaskElasticNet, LogisticRegressionCV)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
                              RandomTreesEmbedding, StackingClassifier, VotingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from datetime import timedelta
import pandas as pd
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import os
import numpy as np
from django.utils import timezone
from time import sleep

simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore")
import sklearn.exceptions

os.environ["PYTHONPATH"] = os.path.dirname(sklearn.exceptions.__file__)


class Vectorizer:
    def __init__(self):
        self.vec1 = CountVectorizer()
        self.vec2 = CountVectorizer()
        self.vec3 = CountVectorizer()
        self.experience_dummies_columns = [f'experience{i/1}' for i in range(100)]

    def fit(self, x, y=None):
        self.vec1.fit(x['question_text'])
        self.vec2.fit(x['answer_text'])
        self.vec3.fit(x['association_text'])
        return self

    def transform(self, x):
        x_copy = x.copy()
        experience_dummies = pd.get_dummies(x_copy['experience']).add_prefix('experience')
        x_copy[self.experience_dummies_columns] = 0
        x_copy[experience_dummies.columns] = experience_dummies.T.replace(False, method='bfill').T
        x_copy.drop(['experience'], axis=1, inplace=True)
        x_copy = pd.concat([x_copy, pd.DataFrame(self.vec1.transform(x_copy['question_text']).toarray(), index=x_copy.index)], axis=1)
        x_copy = pd.concat([x_copy, pd.DataFrame(self.vec2.transform(x_copy['answer_text']).toarray(), index=x_copy.index)], axis=1)
        x_copy = pd.concat([x_copy, pd.DataFrame(self.vec3.transform(x_copy['association_text']).toarray(), index=x_copy.index)], axis=1)
        x_copy.drop(['question_text', 'answer_text', 'association_text'], inplace=True, axis=1)
        x_copy['immediately'] = 0
        x_copy['after10m'] = 0
        x_copy['after_day'] = 0
        x_copy['after_week'] = 0
        x_copy['after_month'] = 0
        x_copy['after_6months'] = 0
        x_copy.loc[x_copy['time_diff_min']<1, 'immediately'] = 1
        x_copy.loc[x_copy['time_diff_min']>10, 'after10m'] = 1
        x_copy.loc[x_copy['time_diff_min']>60*24, 'after_day'] = 1
        x_copy.loc[x_copy['time_diff_min']>60*24*7, 'after_week'] = 1
        x_copy.loc[x_copy['time_diff_min']>60*24*30, 'after_month'] = 1
        x_copy.loc[x_copy['time_diff_min']>60*24*30*6, 'after_6months'] = 1
        x_copy.columns = x_copy.columns.astype(str)
        return x_copy


PROBABILITY_OF_SUCCESS = 0.5
model = GradientBoostingClassifier(warm_start=True)
clf = make_pipeline(Vectorizer(), model)


def get_cards_to_learn():
    start_time = time.time()
    logs = pd.DataFrame(CardLog.objects.all().values())
    logs.replace('<br>', ' ', inplace=True)
    data_X = logs[['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min']]
    data_y = logs['is_good']
    clf.fit(data_X, data_y)
    print(f"fit time={time.time()-start_time}")
    print(f"score {clf.score(data_X, data_y)}")
    cards = Card.objects.all()

    for card in cards:
        pred_time(card)


def pred_time(card):
    for time_q in [timezone.now() - timedelta(days=day)for day in range(7,0,-1)]+[timezone.now() - timedelta(minutes=10), timezone.now() - timedelta(minutes=1), timezone.now(), timezone.now() + timedelta(minutes=1), timezone.now() + timedelta(minutes=10)]+[timezone.now() + timedelta(days=day)for day in range(1,8)]:
        to_pred = pd.DataFrame([[card.question_text, card.answer_text, card.association_text, card.experience,
                                     card.last_remember_min(when=time_q)]],
                                   columns=['question_text', 'answer_text', 'association_text', 'experience',
                                            'time_diff_min'])
        to_pred.replace('<br>', ' ', inplace=True)
        pred = clf.predict_proba(to_pred)[:, 1]
        if pred<PROBABILITY_OF_SUCCESS:
            print(card.question_text)
            print(card.question_text)
            print(pred)
            print(time_q)
            kwargs_card = {'time_to_learn': time_q}
            c = Card.objects.filter(Q(id__icontains=card.id))
            c.update(**kwargs_card)
            break
    else:
        kwargs_card = {'time_to_learn': timezone.now()+ timedelta(days=8)}
        c = Card.objects.filter(Q(id__icontains=card.id))
        c.update(**kwargs_card)


current_learning_date = timezone.now()
get_cards_to_learn()


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
        card = data.iloc[i]
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
    while True:
        saved_card = Card.objects.filter(
            Q(id__icontains=card.id)
        )[0]
        if new_experience == saved_card.experience:
            break

    pred_time(card)
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
        global current_learning_date
        if timezone.now()-current_learning_date>timedelta(days=1):
            get_cards_to_learn()
            current_learning_date = timezone.now()
            print('learned')
        context = super().get_context_data(**kwargs)
        cards_to_remember = {}
        for deck in context['decks']:
            cards = Card.objects.filter(Q(deck=deck))
            if not cards:
                continue
            days = []
            for day in range(7):
                cards_to_learn = 0
                now = timezone.now()
                for card in cards:
                    if card.time_to_learn - now < timedelta(days=0):
                        cards_to_learn+=1
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
        global current_learning_date
        if timezone.now()-current_learning_date>timedelta(days=1):
            get_cards_to_learn()
            current_learning_date = timezone.now()
            print('learned')

        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))

        cards_to_learn = []
        now = timezone.now()
        for card in cards:
            if card.time_to_learn-now< timedelta(days=0):
                cards_to_learn.append(card)
        del card
        if not cards_to_learn:
            return redirect('decks')
        return super(LearnView, self).get(*args, **kwargs)

    def get_queryset(self):
        global current_learning_date
        if timezone.now()-current_learning_date > timedelta(days=1):
            get_cards_to_learn()
            print('learned')
            current_learning_date = timezone.now()

        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))

        cards_to_learn = []
        card_to_learn = None
        now = timezone.now()
        lowest_time_to_remember = timedelta(days=100)
        for card in cards:
            if card.time_to_learn - now < timedelta(days=0):
                cards_to_learn.append(card)
                if lowest_time_to_remember>now - card.time_to_learn:
                    card_to_learn = card
                    lowest_time_to_remember = now - card.time_to_learn
        cards_to_remember = len(cards_to_learn)
        print(lowest_time_to_remember)
        return card_to_learn, cards_to_remember, str(lowest_time_to_remember)


def learn_model(request):
    logs = pd.DataFrame(CardLog.objects.all().values())
    logs.replace('<br>', ' ', inplace=True)
    data_X = logs[['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min']]
    data_y = logs['is_good']
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    models = [LogisticRegression(C=128/(2**i)) for i in range(100)]
    models_working = []
    models_working_diff_max = []
    for model in models:
        clf = make_pipeline(Vectorizer(), model)
        clf.fit(data_X, data_y)
        pred = clf.predict_proba(data_X)
        current_score = pred[:, 1]
        if current_score.max()<PROBABILITY_OF_SUCCESS:
            continue
        best_diff = 0
        for day in range(10):
            test_X = data_X.copy()
            test_X['time_diff_min'] += 1440*(day+1)
            test_X['experience'] += 1
            predtest = clf.predict_proba(test_X)[:, 1]
            diff = (current_score-predtest).min()
            if diff <= 0:
                break
            if best_diff < diff:
                best_diff = diff
            current_score = predtest
            if day == 9:
                models_working.append(model)
                models_working_diff_max.append(best_diff)
                print(model)
                print(pred[:,1].max())
                print(pred[:,1].mean())
                print(pred[:,1].min())
                print(best_diff)
    print(models_working)
    print(np.max(models_working_diff_max))
    print(models_working[np.argmax(models_working_diff_max)])


    # clf = make_pipeline(Vectorizer(), LogisticRegressionCV())
    # clf.fit(X_train, y_train)
    # pred = clf.predict(X_test)
    # print(y_test.mean())
    # print(y_train.mean())
    # 4. print the classfication report
    # print(clf.score(X_test, y_test))
    # print(classification_report(predtest, data_y))
    return redirect('decks')


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