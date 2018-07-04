# import modules and set up logging
import gensim
from gensim.models import KeyedVectors, Word2Vec, word2vec
import logging

# load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# sentences = word2vec.Text8Corpus('text8')
# model = word2vec.Word2Vec(sentences, size=200)

# model.save('text8.model')
# # store the learned weights, in a format the original C tool understands
# model.wv.save_word2vec_format('text8.model.bin', binary=True)
model = Word2Vec.load('text8.model')
# # or, import word weights created by the (faster) C word2vec
# # this way, you can switch between the C/Python toolkits easily
# model = KeyedVectors.load_word2vec_format('vectors.bin', binary=True)


print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5))
# # "boy" is to "father" as "girl" is to ...?
print(model.most_similar(['girl', 'father'], negative=['boy'], topn=5))
# [('mother', 0.61849487), ('wife', 0.57972813), ('daughter', 0.56296098)]
more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))

