# this script is for:
# 1. Tokenization.
# 2. Lowercasing everything.
# 3. Stemming using Porter's algorithm.
# 4. Stopword removal.
# 5. Save the result as a txt file. 

# import module

import sys
import nltk
nltk.download('stopwords') # download stopwords lexion
nltk.download('punkt')     # download tokenize related tools 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from num2words import num2words

# read stopwords
stopwordset = set( stopwords.words( 'english' ) )
stopwordset.update( { num2words( i ) for i in range( 0, 20 ) } ) # update english numbers to stopwords list  

# read file from comment line
with open( sys.argv[1], 'r' ) as f:
    string = f.read()
    f.close()


# function for tokenize -> remove stopwords -> stem
def Preprocessing( sentence ):
    
    # initialize porter stemmer
    stemmer = PorterStemmer()
    
    # tokenize
    words = word_tokenize( sentence )
 
    # remove stopwords and stemming
    words = [ stemmer.stem( w.lower() ) for w in words if w.isalpha() and w.lower() not in stopwordset ] 
    
    return words


# write to new file
with open( sys.argv[2], 'w' ) as f:
    f.write( ' '.join( Preprocessing( string ) ) )
    f.close()
