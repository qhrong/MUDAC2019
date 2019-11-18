# Elmo for word embedding
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
soybean_df = pd.read_csv('soybean_tokens.csv')
tokens_input = list(soybean_df['Tweet'])

tokens_list = []
for ele in tokens_input:
    ele = ele[1:-1].split(", ")
    ele = [i.replace("'", '') for i in ele]
    tokens_list.append(ele)
tokens_list_orig = tokens_list
tokens_length = [len(ele) for ele in tokens_list]
max_len = max([len(ele) for ele in tokens_list])

for ele in tokens_list:
    if len(ele) < max_len:
      ele.extend('' for _ in range(max_len - len(ele)))

# ELMo
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(
    inputs={
        "tokens": tokens_list,
        "sequence_len": tokens_length
    },
    signature="tokens",
    as_dict=True)["elmo"]

print(embeddings.shape)
# The first dimension represents the number of training samples
# The second dimension represents the maximum length of the longest string in the input list of strings
# The third dimension is the length of the ELMo vector
# Hence, every word in the input sentence has an ELMo vector of size 1024

# Extract ELMo vectors
# to arrive at the vector representation of an entire tweet, we will take the mean of the ELMo vectors of constituent terms or tokens of the tweet

def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
  with tf.session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #return sess.run(tf.reduce_mean(embeddings,1))
    message_embeddings = sess.run(embeddings)
    print(message_embeddings)