# politicallean

Attempts to figure out the political leaning (left or right) of specific tweets.  I used twitters api and a list of Senator's twitter handles to construct a dataset of ~300k tweets.  The script "make_datasets.py" can be used to reconstruct it. 

Note: to do this, you need to replace the values for 'consumer_key', 'consumer_secret', access_token_key', and 'access_token_secret' with your own twitter api access values.

You also need to get a copy of the GloVe word embedding, which can be found at https://nlp.stanford.edu/projects/glove/

I originally ran this on my laptop, but ran into performance issues, so I ported the project over to Google Colab (which is where the Jupyter notebook comes from).
