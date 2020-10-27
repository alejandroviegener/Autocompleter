# Autocompleter / NLP / ML Engineering Challenge

Autocompleter implemented using an n gram language model


Motivation
----------

Your challenge is to design and implement an auto-complete server using the sample chat histories in sample_conversations.json. The basic idea is for the agent to type a few letters or words, and the service will suggest sentence completions. 

Usage
-----

- install required packages with:
   pip install -r requirements.txt

- Build and serialize your Autocomplete object using:
   python autocomplete_build.py

- You can run your serialized Autocomplete server on localhost:5000 by executing:
   python autocomplete_server.py

- You can query the running server with:
   curl localhost:5000/autocomplete?q=what+is+your

- You can run the tests with:
   pytest


