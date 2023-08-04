# stockLang_model
Large Language model for stock prediction

This is an exercise for sotck market proce mode predictiom based on kind of language model
It is inspired by Andrej's Karpathy course - "NN_zero_to_hero" and right now this very first attempt
to calculate price candles pairs, specifically for NQ futures (NASDAQ 100)

Hopefully it will move forward from pure mechanical calculation to NN based approach.
The idea and the plan roughly are following
 - create some kind of candle dictionaly to construct artificial candles series
 - mode a languange model to predict only next candle in pair but also next candle based on candles series preceding to a new one.
The Idea actually based on multy year of price action observation on futures/forext markets and cler inderstanding that there are some
repetitive patters which price is drawing while move up and down. Obvious that this is not such a simple, but should be good anough at least as
DL modeling and programming practice.

Very first version of Jupyter notebook and NQ price dataset is uploaded.
