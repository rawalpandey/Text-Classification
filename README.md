# Text-Classification
### 1.Introduction
This text classification trains a recurrent neural network on the IMDB large movie review dataset for
sentiment analysis . We want to develop an application which will predict the text based on sentimental
analysis depending upon the reputable information provided previously.
### OVERVIEW
The topic of text classification prediction has recently attracted tremendous attention. The
basic countermeasure of comparing websites against a list of labeled reviews sources is inflexible, and
so in deep learning we use recurrent neural network approach which is the best suited algorithm. Our
project aims to use RNN and LSTM to perform the text classification directly, based on the sentiment
analysis.
### BACKGROUND AND MOTIVATION
Nowadays, information technology is developing fast and used in more and more fields.
Therefore, numerous documents are saved in the computers which could be read by the computers.
Moreover the number of the documents increases extremely everyday. For the important application,
it becomes a research subject how to automatically classify, organize and manage such numerous
amount of literature and data, in the case most of them are documents. Due to the growing availability
of digital textual documents, automatic text classification (ATC) has been actively studied to organize
a vast number of unstructured documents into a set of categories, based on the textual contents of the
document. Text representation is the fundamental step in text classification task, in which a text is
represented by a set of features. Features play the important roles in training classification model and
prediction. Many previous studies focused on enriching text representation to address text
classification task. However, the traditional classification methods with RNN only studied intensively
on the words and their relationship in some specific corpus/dataset.
### OBJECTIVE

. Our focus is to develop an application to classify the review of movies dataset using RNN. We
intend to use LSTM to classify the polarity of reviews into positive and negative.

### METHODOLOGY
We have used a RNN using LSTM to classify the text and movie review articles to build a classifier that
can make decisions about information based on the content from the corpus. </br>The model focus on
classifying the text based on the semantic analysis. 
Detailed process of how model is built is explained further in the report.</br>
Basic Methodology can be described in following steps.</br>
###### Step 1 : Collecting Data</br>
We are using tensorflow dataset of imdb reviews. The IMDB large movie review dataset is a
binary classification dataset—all the reviews have either a positive or negative sentiment. The
dataset will be imported from tensorflow-datasets library.</br>
###### Step 2: Data preprocessing</br>
A simple approach is to assume that the smallest unit of information in a text is the word (as
opposed to the character). Therefore, we will be representing our texts as word sequences.</br>
The sentence is “ This is a Cat.”</br>
In this example, we removed the punctuation and made each word lowercase because we assume
that punctuation and letter case don’t influence the meaning of words. In fact, we want to avoid
making distinctions between similar words such as This and this or cat. and cat.
The sequence of words are converted and encoded into integers using tensorflow encoder. This
encoded integer is dictionary of words with unique integer assigned to each word.</br>
###### Step 3: Prepare the data for training</br>
Next create batches of these encoded strings. Use the padded_batch method to zero-pad the
sequences to the length of the longest string in the batch:</br>
BUFFER_SIZE = 10000</br>
BATCH_SIZE = 64</br>
train_dataset = train_dataset.shuffle(BUFFER_SIZE)</br>
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)</br>
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)</br>
###### Step 4: Create the model</br>
Build a tf.keras.Sequential model and start with an embedding layer. An embedding layer stores
one vector per word. When called, it converts the sequences of word indices to sequences of
vectors. These vectors are trainable. After training (on enough data), words with similar meanings
often have similar vectors.</br>
This index-lookup is much more efficient than the equivalent operation of passing a one-hot
encoded vector through a tf.keras.layers.Dense layer.A recurrent neural network (RNN) processes sequence input by iterating through the elements.
RNNs pass the outputs from one timestep to their input—and then to the next.</br>
The tf.keras.layers.Bidirectional wrapper can also be used with an RNN layer. This propagates the
input forward and backwards through the RNN layer and then concatenates the output. This helps
the RNN to learn long range dependencies.</br>
For LSTM feature extraction is done by text encoding and embedding. How it is done is explained
in tool description.</br>
###### Step 5: Train the model</br>
history = model.fit(train_dataset, epochs=10,
validation_data=test_dataset,
validation_steps=30)</br>
The model is trained with the input obtained from text encoding and embedding. Once the model
is trained, we can validate it by calculating its accuracy.</br>
Once we get the model ready, we can use to develop the application.
### 2.Tool Description
##### TENSORFLOW:-
The most famous deep learning library in the world is Google's TensorFlow. Google product uses
machine learning in all of its products to improve the search engine, translation, image captioning or
recommendations.
Tensorflow architecture works in three parts:</br>
- Preprocessing the data</br>
- Build the model</br>
- Train and estimate the model</br>
It is called Tensorflow because it takes input as a multi-dimensional array, also known as tensors.
The input goes in at one end, and then it flows through this system of multiple operations and
comes out the other end as output.</br>
A tensor can be originated from the input data or the result of a computation. In TensorFlow, all
the operations are conducted inside a graph. The graph is a set of computation that takes place
successively. Each operation is called an op node and are connected to each other.</br>
###### Feature Extraction - Word Embedding
We cannot use the Doc2Vec for preprocessing because it will transfer the entire document
into one vector and lose the order information. To prevent that, we use the word embedding
instead. We first clean the text data by removing all characters which are not letters nor numbers.
Then we count the frequency of each word appeared in our training dataset to find 5000 most
common words and give each one, a unique integer ID. For example, the most common word will
have ID 0, and the second most common one will have 1, etc. After that we replace each common
word with its assigned ID and delete all uncommon words. Notice that the 5000 most common
words cover the most of the text, as shown in Figure 1, so we only lose little information but
transfer the string to a list of integers. Since the LSTM unit requires a fixed input vector length,
we truncate the list longer than 500 numbers because more than half of the news is longer than
500 words. Then for those lists shorter than 500 words, we pad 0’s at the beginning of the list.We also delete the data with only a few words since they don’t carry enough information
for training. By doing this, we transfer the original text string to a fixed length integer vector while
preserving the words order information. Finally, we use word-embedding to transfer each word ID
to a 32-dimension vector.</br>
The word embedding will train each word vector based on word similarity. If two words
frequently appear together in the text, they are thought to be more similar and the distance of their
corresponding vectors is small. The pre-processing transfers each news in raw text into a fixed
size matrix.
###### Semantic Analysis

Text classification offers a good framework for getting familiar with textual data processing
without lacking interest, either. In fact, there are many interesting applications for text classification
such as spam detection and sentiment analysis. In this post, we will tackle the latter and show in
detail how to build a strong baseline for sentiment analysis classification. This will allow us to get
our hands dirty and learn about basic feature extraction methods which are yet very efficient in
practice.</br>
Sentiment analysis aims to estimate the sentiment polarity of a body of text based solely
on its content. The sentiment polarity of text can be defined as a value that says whether the
expressed opinion is positive (polarity=1), negative (polarity=0), or neutral. In this tutorial, we
will assume that texts are either positive or negative, but that they can’t be neutral.</br>
###### Prediction Model - Neural Network Embeddings
An embedding is a mapping of a discrete — categorical — variable to a vector of continuous
numbers. In the context of neural networks, embedding’s are low-dimensional, learned continuous
vector representations of discrete variables. Neural network embedding are useful because they
can reduce the dimensionality of categorical variables and meaningfully represent categories in
the transformed space.</br>
Neural network embedding have 3 primary purposes:</br>
1. Finding nearest neighbors in the embedding space. These can be used to make
recommendations based on user interests or cluster categories.
2. As input to a machine learning model for a supervised task.</br>
3. For visualization of concepts and relations between categories.</br>
LSTM – Long short-term memoryLong short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture
used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has
feedback connections. Sequence prediction problems have been around for a long time. They are
considered as one of the hardest problems to solve in the data science industry. These include a
wide range of problems; from predicting sales to finding patterns in stock markets’ data, from
understanding movie plots to recognizing your way of speech, from language translations to
predicting your next word on your iPhone’s keyboard.</br>
LSTMs have an edge over conventional feed-forward neural networks and RNN in many ways.
This is because of their property of selectively remembering patterns for long durations of time.
The purpose of this article is to explain LSTM and enable you to use it in real life problems.
