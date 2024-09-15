from django.urls import path

from . import views

urlpatterns = [
    path('add_deck_form', views.add_deck_form, name='add_deck_form'),
    path('add_card_form', views.add_card_form, name='add_card_form'),
    path('', views.IndexView.as_view(), name='decks'),
    path('cards', views.CardsView.as_view(), name='cards'),
    path('delete_deck/<str:deck_name>/', views.deck_delete, name='deck_delete'),
    path('delete_card/<str:card_name>/', views.card_delete, name='card_delete'),
    path('edit_card', views.EditView.as_view(), name='edit_card'),
    path('edit_card_form', views.edit_card_form, name='edit_card_form'),
    path('log_card', views.LearnView.as_view(), name='log_card'),
    path('log_card_form', views.log_card_form, name='log_card_form'),
    path('data_upload', views.data_upload, name='data_upload'),
]