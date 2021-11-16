from django.shortcuts import render
from rest_framework.decorators import api_view
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from . serializers import *

from django.views import View

from .apps import DronaConfig

from .Medical_InsuranceClaim import model_lr

import numpy as np
import nltk
import random


# Create your views here.
class dronaList(APIView, View,):
    serializer_class = dronaSerializer

    def get(self, request):
        # detail = [{"name": detail.inputs}
        detail = [{"name": dronaList.chatbot_response(detail.inputs)}
                  for detail in dronaList.objects.all()]
        # print(employeeList.chatbot_response('hello'))

        return Response(detail)

    def post(self, request):

        serializer = dronaSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [DronaConfig.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = dronaList.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))

    def predict_class(sentence, model):
        # filter out predictions below a threshold
        p = dronaList.bow(sentence, DronaConfig.words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": DronaConfig.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(msg):
        ints = dronaList.predict_class(msg, DronaConfig.model)
        res = dronaList.getResponse(ints, DronaConfig.intents)
        return res

    # def final(request):
    #     if request.method == 'POST':
    #         testsss = request.POST['testsss']
    #         data = dronaList.chatbot_response(testsss)
    #         # data = {testsss:}
    #         # print(testsss)
    #         # , {'datas': data}
    #         print(dronaList.chatbot_response(testsss))
    #     return render(request, "index.html", {'datas': data})

    def final(request):
        if request.method == 'POST':
            testsss = request.POST['testsss']
            data = model_lr(testsss)
            # data = {testsss:}
            # print(testsss)
            # , {'datas': data}
            # print(dronaList.chatbot_response(testsss))
        # return render(request, "index.html", {'datas': data})
        return render(request, "index.html", {'datas': data})

# def home(request):
#     return render(request, 'home.html')


