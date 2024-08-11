# Generated by Django 4.2.14 on 2024-08-08 18:20

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Card',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_text', models.CharField(max_length=1000)),
                ('answer_text', models.CharField(max_length=1000)),
                ('association_text', models.CharField(max_length=1000)),
                ('last_remembered', models.DateTimeField(default=None, null=True, verbose_name='last remembered')),
                ('experience', models.FloatField(default=0)),
            ],
        ),
    ]