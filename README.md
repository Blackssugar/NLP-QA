# NLP-QA

NLP中的机器阅读理解,能根据问题找到答案在文档中的位置

<img src="https://github.com/Blackssugar/NLP-QA/blob/master/architecture.png" width=500/>

## 1.WordEmbedding:

For word embedding, I used pre-trained word2vecmodel glove-wiki-gigaword-100. However, since there are a lot of symbols and numbers inside my vocabulary, when creating the embedding matrix for the embedding layer, I found almost half of the vocabulary does not have embedding in this model. I then trained my own Skip Gram model using Gensim library to provide embeddings for words that are not in the pre-trained model. The output shape of each document is then (1018,100) in which 1018 is the maximum length of the documents, and 100 is the embedding size for each word.

### TF-IDF: 

I used the TfidfVectorizer from the sklearn library. The input for the vectorizer is the document text of both train and test data. The output shape for each document is a (257,) vector. I then pad and reshape the vectors to (1018,1), so the vectors can be concatenated with the word2vec vectors.

### PoS Tags: 

For Pos Tags, I used nltk.pos_tag to tag the document sentences separately. At the beginning, I tried to use NLTK’s another package called HiddenMarkovModelTrainer, however, it is really slow in turns of tagging sentences, I then switched to nltk.pos_tag. I also created a tag dictionary to assign distinct numerical value for each tag, so I can convert the tags into a number. As I did for TF-IDF vectors, I pad and reshape the output of each PoS vector into (1018,1).

### Name Entity Tags: 

For Name Entity Tags, I used spacy model ‘en_core_web_sm’ to extract the name entities from each document. Same as what I did for above features, I created a dictionary to assign distinct values for each tag, to convert the tags into numbers. Then pad and reshape each vector into (1018,1).

### Dependency Path: 

For dependency path, I used spacy model ‘en_core_web_sm’ to extract the dependency for each document. The are two values that I can extract, dependency relations and head. I choose to use the head values, since I do not need to convert the values into numbers as they already are. Then pad and reshape each vector into (1018,1).


## 2. SequenceModel(RNNwithAttention):

ThesequencemodelIimplementedis Bi-LSTM with attention. There are five input layer for document part, each represent a feature of document data, i.e: word embedding, TF-IDF, Pos tags, Name Entity tags, Dependency path. There is one input layer for question part, represents input for word embedding. There is a embedding layer in both document and question parts. The embedding layer is used for embed the word tokens into word2vec vectors. For the other four features in the document part, I extracted and reshaped them before construct the model, therefore no embedding layers are implemented for the four features. There are two Bi-LSTM for document input and question input respectively. In the document part, the output from word embedding layer is concatenated with the four feature inputs. The concatenated output is them put into the Bi-LSTM layer. In the question part, the output from word embedding layer is put into the Bi-LSTM layer. The Bi-LSTM is then connect to a flatten layer follow by a dense layer to extract a question summary. The attention layer is implemented afterwards. It computes the dot products of the hidden states from the Bi-LSTM in the document side and the question summary from the question side. Softmax is then applied after the attention layer. The argmax is extracted as the answer. In this case, the answer is the sentence id in the document that has the answer to the question.
