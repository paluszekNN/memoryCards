from django.urls import path

from . import views

urlpatterns = [
    path('add_deck', views.add_deck, name='add_deck'),
    path('add_deck_form', views.add_deck_form, name='add_deck_form'),
    path('', views.IndexView.as_view(), name='decks'),
    path('cards', views.CardsView.as_view(), name='cards'),
]