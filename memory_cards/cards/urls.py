from django.urls import path

from . import views

urlpatterns = [
    path('add_deck_form', views.add_deck_form, name='add_deck_form'),
    path('add_card_form', views.add_card_form, name='add_card_form'),
    path('', views.IndexView.as_view(), name='decks'),
    path('cards', views.CardsView.as_view(), name='cards'),
    path('delete_deck/<str:deck_name>/', views.deck_delete, name='deck_delete'),
]