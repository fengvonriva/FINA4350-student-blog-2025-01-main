---
Title: Decoding Crypto Volatility: A New Direction
Date: 2025-05-06 23:59
Category: Reflective Report
Tags: Group DeepText Analysts
Slug: DeepText Analysts
---

# Our New Direction

As we progress from the initial phase of our project, we began with the goal of developing a model to predict cryptocurrency prices. However, we encountered several challenges that highlighted the limitations of this approach which included the objective definition of sentiment, whether correlation necessarily exists between the sentiment on Reddit and prices, and most importantly, whether our model actually identifies and aggregates the sentiment correctly in the first place. Hence, after careful consideration, we’ve decided to shift our focus towards utilizing the model exclusively for sentiment analysis and deriving sentiment scores. This new direction will allow us to achieve more reliable and meaningful insights. 

Our new idea, as visualised in Figure 1, involves using different models such as dictionary models and deep learning models to generate our sentiment scores and with those scores, we will move on to build a trading simulator. Based on the sentiment score obtained for each day, the trading simulator will make a buy or sell decision. This will allow us to test out different investment strategies and evaluate whether we make profits or losses. Moving forward, we plan to build an application that can provide users a simpler and more accurate way of investing in cryptocurrencies based on sentiment analysis of Reddit comments.

![Picture showing application structure]({static}/images/DeepText-Analysts-02-application-structure.jpg)

First thoughts by Feng:

1) Goal in the beginning: predicting prices successfully

2) Realization in the meantime: how About we first try to capture the Sentiment accurately?