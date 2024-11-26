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
from sklearn.linear_model import LogisticRegression
from datetime import timedelta
import pandas as pd
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import os
import numpy as np

simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore")
import sklearn.exceptions

os.environ["PYTHONPATH"] = os.path.dirname(sklearn.exceptions.__file__)
PROBABILITY_OF_SUCCESS = 0.8
model = LogisticRegression(C=3.814697265625e-06)


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
        logs = pd.DataFrame(CardLog.objects.all().values())
        logs.replace('<br>', ' ', inplace=True)
        data_X = logs[['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min']]
        data_y = logs['is_good']
        clf = make_pipeline(Vectorizer(), model)
        clf.fit(data_X, data_y)

        context = super().get_context_data(**kwargs)
        cards_to_remember = {}
        for deck in context['decks']:
            cards = Card.objects.filter(Q(deck=deck))
            days = []
            for day in range(7):
                cards_to_learn = 0
                to_pred = pd.DataFrame(columns=['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min'])
                for card in cards:
                    to_pred = pd.concat([to_pred, pd.DataFrame([[card.question_text, card.answer_text, card.association_text, card.experience,
                                                                 card.last_remember_min(when=timezone.now() + timedelta(days=day))]],
                                           columns=['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min'])])
                to_pred.replace('<br>', ' ', inplace=True)
                pred = pd.Series(clf.predict_proba(to_pred)[:, 1])
                pred = list(pred.loc[pred < PROBABILITY_OF_SUCCESS].index)
                for ind in pred:
                    card = cards[ind]
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
        logs = pd.DataFrame(CardLog.objects.all().values())
        logs.replace('<br>', ' ', inplace=True)
        data_X = logs[['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min']]
        data_y = logs['is_good']
        clf = make_pipeline(Vectorizer(), model)
        clf.fit(data_X, data_y)

        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))
        cards_to_remember = []
        for log in CardLog.objects.all():
            if log.time_diff_min<0:
                print(log.id)
        print("cards")
        for card in Card.objects.all():
            if card.last_remember_min()<0:
                print(card.last_remember_min())
                print(card.id)

        to_pred = pd.DataFrame(
            columns=['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min'])
        for card in cards:
            to_pred = pd.concat(
                [to_pred, pd.DataFrame([[card.question_text, card.answer_text, card.association_text, card.experience,
                                         card.last_remember_min(when=timezone.now())]],
                                       columns=['question_text', 'answer_text', 'association_text', 'experience',
                                                'time_diff_min'])])
        to_pred.replace('<br>', ' ', inplace=True)
        pred = pd.Series(clf.predict_proba(to_pred)[:, 1])
        pred = list(pred.loc[pred < PROBABILITY_OF_SUCCESS].index)
        del card
        if not pred:
            return redirect('decks')
        return super(LearnView, self).get(*args, **kwargs)

    def get_queryset(self):
        logs = pd.DataFrame(CardLog.objects.all().values())
        logs.replace('<br>', ' ', inplace=True)
        data_X = logs[['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min']]
        data_y = logs['is_good']
        clf = make_pipeline(Vectorizer(), model)
        clf.fit(data_X, data_y)

        deck = Deck.objects.filter(Q(id__icontains=self.request.GET.get('q')))[0]
        cards = Card.objects.filter(Q(deck=deck))
        card_to_remember = None
        lowest_time_to_remember = PROBABILITY_OF_SUCCESS
        cards_to_remember = 0

        to_pred = pd.DataFrame(
            columns=['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min'])
        for card in cards:
            to_pred = pd.concat(
                [to_pred, pd.DataFrame([[card.question_text, card.answer_text, card.association_text, card.experience,
                                         card.last_remember_min(when=timezone.now())]],
                                       columns=['question_text', 'answer_text', 'association_text', 'experience',
                                                'time_diff_min'])])
        to_pred.replace('<br>', ' ', inplace=True)
        pred = pd.Series(clf.predict_proba(to_pred)[:, 1])
        pred = pred.loc[pred < PROBABILITY_OF_SUCCESS]
        pred_ind = list(pred.index)
        cards_to_remember = pred.count()
        card_to_remember = cards[pred_ind[int(np.argmax(pred))]]
        lowest_time_to_remember = pred[pred_ind[int(np.argmax(pred))]]
        print(lowest_time_to_remember)
        return card_to_remember, cards_to_remember, str(lowest_time_to_remember)


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.linear_model import (LogisticRegression, Lasso, RidgeClassifier, MultiTaskLasso, LassoLars, LassoLarsIC, PassiveAggressiveClassifier, SGDClassifier, Perceptron, ElasticNet,
                                  OrthogonalMatchingPursuit,BayesianRidge,MultiTaskElasticNet)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
                              RandomTreesEmbedding, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


class Vectorizer:
    def __init__(self):
        self.vec1 = CountVectorizer(ngram_range=(3,3))
        self.vec2 = CountVectorizer(ngram_range=(3,3))
        self.vec3 = CountVectorizer(ngram_range=(3,3))
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
        x_copy[experience_dummies.columns] = experience_dummies
        x_copy.drop(['experience'], axis=1, inplace=True)
        x_copy = pd.concat([x_copy, pd.DataFrame(self.vec1.transform(x_copy['question_text']).toarray(), index=x_copy.index)], axis=1)
        x_copy = pd.concat([x_copy, pd.DataFrame(self.vec2.transform(x_copy['answer_text']).toarray(), index=x_copy.index)], axis=1)
        x_copy = pd.concat([x_copy, pd.DataFrame(self.vec3.transform(x_copy['association_text']).toarray(), index=x_copy.index)], axis=1)
        x_copy.drop(['question_text', 'answer_text', 'association_text'], inplace=True, axis=1)
        return x_copy


def learn_model(request):
    logs = pd.DataFrame(CardLog.objects.all().values())
    logs.replace('<br>', ' ', inplace=True)
    data_X = logs[['question_text', 'answer_text', 'association_text', 'experience', 'time_diff_min']]
    data_y = logs['is_good']
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    models = [LogisticRegression(C=128/(2**i)) for i in range(100)]
    models_working = []
    models_working_diff_max= []
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