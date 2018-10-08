# TalkSick

Violence in social media is something that we all deal with on a daily basis, and I would love to see less off it on my screen.

This is a project I did at Metis data science immersive program.
My motivation for this project was promoting a positive approach to minimize intentional and unintentional violent conversations taking place online by classifying toxic comments in social networks.

And so I built a user-friendly and reasonably **smart Slack bot that warns users and suggests to reconsider their words**.

# Contents

[**1. Why a smart policing bot?**](#why_bot)

[**2. Data**](#get_data)

[**3. TalkSick Model**](#model)

[**4. Demo: TalkSick in action**](#demo)



[**5. How smart is the bot?**](#bot_smart)

---

# <a name="why_bot">1. Why a smart policing bot?</a>

[Slack](https://slack.com/) is a popular messaging app for teams that allows team members to discuss different topics, organized in channels. Now, when a new member joins a Slack team that's been around for some time, they sometimes tend to post messages in wrong channels. Nobody wants to be that nagging senior team member trying to direct this rookie to a more appropriate place. 

Wouldn't it be nice to have a smart bot that can learn topics of different channels, monitor the discussions and then warn users if they go off topic? Building such a bot is the aim of this project.

---

# <a name="get_data">2. Data</a>

In order to build my bot and see how accurately it's performing, I needed some data. Slack data is hard to come by, since it's private. The next best thing is [Reddit](https://www.reddit.com/), since its data is easily available and has a similar structure to Slack. 

For the purpose I used Toxic Comment Classification dataset from Kaggle to train a model.
You can fetch the dataset [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).

I used Fasttext word embeddings:

* [FastText crawl 300d 2M](https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m)

---

# <a name="model">4. TalkSick Model</a>

Now that we've seen the bot in action, let's see how does it do all this. 

The bot obviously needs to be able to compare two messages (more generally, documents): the user's input and one of the messages already present in a channel. A standard way to compare two documents is to use the [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW) approach, which assigns a sparse vector to each document, with elements related to the number of times a given word appears in the document (this includes approaches such as [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) as well). Once such document vectors are generated, the similarity of the two documents is measured by calculating the cosine between the corresponding vectors: higher cosine similarity indicates more similar documents. 

<img align="right" width="200px" src="images/two_documents.png" hspace="20" vspace="20">
However, problems arise when two documents share no common words, but convey similar meaning, such as in the example on the right. The BoW / tf-idf vectors of these two documents are perpendicular, yielding zero cosine similarity. 



## 4.1 Word Mover's Distance

A way to fix this was proposed recently at the *International Conference on Machine Learning* in 2015 in the form of [Word Mover's Distance](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf) (WMD) model. This model starts by representing each of the words in a document with vectors in a high dimensional (word embedding) vector space by using [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (w2v), a popular neural network model trained to reconstruct the semantic context of words. What that essentially means is that the words that have similar meaning will have their w2v vectors close to each other, as illustrated below. I will use a pre-trained word2vec model from the [Spacy](https://spacy.io/) package, which has been trained on the Common Crawl corpus.

<img align="right" width="300px" src="images/w2v_space.png" hspace="20" vspace="20">
Then, a natural way to estimate how dissimilar, or distant, the two documents are is to look at the distance between the corresponding word vectors and, roughly speaking, add those distances up. This metric is called the Word Mover's Distance, because it is an instance of the well-known [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover's_distance) (EMD) optimization problem, only formulated in the word embedding space. 

What follows is an intuitive, but somewhat technical explanation of the EMD problem, important for understanding the modification of the WMD that I'll have to implement. <a onclick="showhide('expl1')">Click here for more details.</a>
<div style="display:none" id="expl1">
The EMD assumes that one has two collections of vectors, let's call them the <i>senders</i> and the <i>receivers</i>, and a matrix of their pair-wise distances. In addition to this, each of the vectors has a weight, which is a real number smaller than 1, indicating how much "goods" each of the sender vectors has to send and how much of the goods each of the receiver vectors needs to receive. The sum of all the weights for the receiver vectors is normalized to 1, as it is for the sender vectors. The problem now is, given the distances (costs) between the sender-receiver pairs, to determine the most efficient way to <i>move</i> the goods from the senders to the receivers, allowing for partial sending and receiving (i.e. so that a sender can send a portion of its goods to one receiver and another portion to another receiver). This is obviously a non-trivial constrained optimization problem, and it has a known solution, which can be easily implemented in Python with the <a href="https://pypi.python.org/pypi/pyemd"><code class="highlighter-rouge">pyemd</code></a> package. <br><br>

WMD is the application of the EMD problem to the context of word embeddings, where the senders and receivers are w2v vectors of words from the first and second document we're comparing, respectively. The weights of the vectors are chosen to be proportional to the number of times the corresponding word appears in the document. The distances between the vectors are then calculated using standard Euclidean distances in the word embedding space. In this way we can easily calculate the WMD distance between two documents using the <code class="highlighter-rouge">pyemd</code> package.
</div>


## 4.2 Modifying WMD

A practical obstacle in applying this method to our case is the fact that the EMD algorithm has a horrible time complexity: <i>O(p^3 log(p))</i>, where <i>p</i> is the number of unique words in the two documents. We would need to compare the user's input to all of the previous messages in all the channels, calculate the average distance for each of the channels, and the one with the smallest average distance would be our prediction for the channel to which the user's message should go. If the user posted the message in the channel we predicted, the bot doesn't do anything, otherwise the bot will advise the user to consider posting it to the predicted channel. For Slack teams that contain a lot of messages spread out over a lot of channels, this will not be a feasible approach for a real time response of the bot. 

<img align="right" width="300px" src="images/frequencies.png" hspace="20" vspace="20">
So I needed to modify the WMD method somewhat. Comparing the input message to *all* the messages in a given channel seems excessive: surely there are messages that are more "representative" of the channel content than the others, and it's likely enough to compare the user input to those messages only. However, this would require expensive preprocessing, in which we essentially have to sort the channel messages using WMD as a key. But can we somehow *construct* a single message representative of an entire channel? 

Intuitively, this could be achieved by looking at word distributions in a given channel, as shown on the right. Obviously, to a person, looking at the first, say, 10 or so words that occur most often in a channel would give a pretty good idea of what that channel is about. A single message representative of that channel should therefore contain only those 10 words. To use this message in EMD / WMD, we need to choose the weights (see previous subsection) of the vectors representing the words in it. Since the weights in a standard WMD are directly proportional to how many times a given word appears in a message, we can make the weights in our representative message proportional to the number of times a given word appears in the entire channel (and then normalize it). 

In this way we've constructed a single representative message for each channel, and we only need to calculate the WMD distances between the input message and each of the representative messages, find the shortest one, and predict the corresponding channel as the one the input message is supposed to go to.

---

# <a name="bot_smart">5. How smart is the bot?</a>

But is 10 words enough to form a representative message? Is it 30? A 100? We can find the optimal number of words,  <i>n_words</i>, by treating it as a hyperparameter and tuning it on a validation set. Our entire corpus will consist of about 5000 comments of varying length from each of the 5 channels, which will be split into a training (70%), validation (20%) and test (10%) set. 

The training set will be used to infer the word distributions for each of the channels. The text will be pre-processed in a standard way (by removing punctuation, stop words, etc.), and in addition to that we'll also use `Spacy`'s parts-of-sentence tagging to only leave the nouns, verbs and adjectives, as they are the ones carrying most of the meaning. We will also remove all the words that are not in `Spacy`'s pre-trained vocabulary, since we won't have a vector representation for those (luckily, only about 2.5% of the words are not in the vocabulary).

The validation set will be used to determine the best value for <i>n_words</i>, which turned out to be 180.

<img align="right" width="300px" src="images/confusion_matrix.png" hspace="20" vspace="20">
Finally, the test set will be used to compute the final accuracy of our model. On the right we see the confusion matrix for the test set, showing the distributions of messages from their true categories over the predicted ones. The accuracy of this model is about 74%, which is pretty good, and a noticeable improvement from 68% that one gets from the tf-idf approach and using the cosine similarity as the metric. 


---

# <a name="demo">4. Demo: TalkSick in action</a>

TalkSick is a Slack bot, she would target offensive content and privately warn users if their content might be offensive to others.

To showcase my bot's might, I made a demo Slack channel and invited all my classmates to the channel to test her.

To build the bot I used the excellent package [`slackclient`](https://github.com/slackapi/python-slackclient) that allows one to communicate with Slack's API from Python. 

Below is a little illustration of bot's basic functionality, showing me entering a couple of messages in the test channel. As you can see, when I just type generic things she won’t bother me, but if i write something offensive she would warn me with a message that is only visible to me, saying that my content might be offensive and kindly asks me to consider rephrasing.

![](TalkSick.gif)


### Preprocessing

* Spellings correction: by comparing the Levenshtein distance and a lot of regular expressions.
* Custom tokenizer: data-cleaning, spelling correction and translation

### Models


