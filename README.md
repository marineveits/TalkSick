# TalkSick

Violence in social media is something that we all deal with on a daily basis, and I would love to see less off it on my screen.

This is a project I did at Metis data science immersive program.
My motivation for this project was promoting a positive approach to minimize intentional and unintentional violent conversations taking place online by classifying toxic comments in social networks.

And so I built a user-friendly and reasonably **smart Slack bot that warns users and suggests to reconsider their words**.

# Contents

[**1. Why a smart policing bot?**](#why_bot)

[**2. Data**](#get_data)

[**3. Demo: TalkSick in action**](#demo)

[**4. TalkSick Model**](#model)



---

# <a name="why_bot">1. Why a smart policing bot?</a>

[Slack](https://slack.com/) is a popular messaging app for teams that allows team members to discuss different topics, organized in channels. Now, Slack's private channels and messaging content is not being monitored, and so does violent or offensive content might be overlooked. 

Wouldn't it be nice to have a smart bot that can listen to differnt channels' conversasions and warn users with a message that is only visible to them, saying that my content might be offensive and kindly asks me to consider rephrasing? Building such a bot is the aim of this project.

---

# <a name="get_data">2. Data</a>

In order to build my bot and see how accurately it's performing, I needed some data. Slack data is hard to come by, since it's private. The next best thing is [Reddit](https://www.reddit.com/), since its data is easily available and has a similar structure to Slack. 

For the purpose I used Toxic Comment Classification dataset from Kaggle to train a model.
You can fetch the dataset [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
This competition was powered by Jigsaw and Google. The dataset includes about 300K labeled posts from Reddit, with six levels of toxicity: toxic, severe toxic, obscene, threat, insult and identity hate.


---


# <a name="demo">3. Demo: TalkSick in action</a>

TalkSick is a Slack bot, she would target offensive content and privately warn users if their content might be offensive to others.

To showcase my bot's might, I made a demo Slack channel and invited all my classmates to the channel to test her.

To build the bot I used the excellent package [`slackclient`](https://github.com/slackapi/python-slackclient) that allows one to communicate with Slack's API from Python. 

Below is a little illustration of bot's basic functionality, showing me entering a couple of messages in the test channel. As you can see, when I just type generic things she won’t bother me, but if i write something offensive she would warn me with a message that is only visible to me, saying that my content might be offensive and kindly asks me to consider rephrasing.

![](TalkSick.gif)


---


# <a name="model">4. TalkSick Model</a>

Now that we've seen the bot in action, let's see how does it do all this. 

The bot obviously needs to be able to listen to all channel's members' conversations, and send a private warning message to  users if their content might be offensive to others and asks them to reconsider. 

To build and train the model I used a dataset from Kaggle <a name="get_data">.
I used AWS GPU to get the work done. Used  [FastText crawl 300d 2M](https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m), sklearn, tensorflow and keras to build a Kmax Pooling CNN model. And eventually created a Slack bot to target offensive content. 

The data is multiclass and multilabel, so I had to be creative with picking the right techniques. For the data preprocessing part I created a custom tokenizer to account for patterns that people often use when they're typing angrily. I also needed to be able to correct typos in order to implement Fasttext word embedding vectors, so I made a spell checker using Levinstein distances.And finally, I used it all to train a Kmax pooling CNN model, using 10 folds and 20 epochs in each.


To build this model I basically replicated the Kmax pooling CNN structure from this paper: [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf).
and this a basic schema of its structure

The model resulted in a 0.98 AUC score (on accuracy), and I decided to use F5 score which weighs recall higher than precision (by avoiding false negatives), and resulted in 0.89. This metric is appropriate for the case because the bot would only warn users and not block them so the cost of false negative predictions is higher in this case. 




