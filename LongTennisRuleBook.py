# Script to Lab Week 3
import nltk
import re
from nltk.collocations import *
from nltk import FreqDist
import os
os.getcwd()

#file2 = nltk.corpus.gutenberg.fileids( ) [2]
# print " Used file Name == " + file2
#filetext = nltk.corpus.gutenberg.raw(file2)

moviesfile = open('LongTennisRuleBook.txt')
filetext = moviesfile.read()
print len(filetext)
#filetext[1000:1100]
print "***Removing number as subtitles have timestamps***"
filetext = re.sub("\d+", " " , filetext)
print len(filetext)
#print filetext[1000:1100]
print "***Removing punctuations and tokenising the file***"
from nltk.tokenize import RegexpTokenizer
punctuation_tokenizer = RegexpTokenizer(r'\w+')
filetokens = punctuation_tokenizer.tokenize(filetext)
print "Length of file tokens : " + str(len(filetokens))
#print filetokens[1000:10000]

print "=================="

#f = open('C:\\Python27\\texttest.txt')
#filetext = f.read()
#len(filetext)
#filetext = filetext [ 20000:]
#len(filetext)
#filetokens = nltk.word_tokenize(filetext)
filewords = [w.lower( ) for w in filetokens]
#filewords[1000:]
shortwords	=	filewords[20000:150000]
shortwords

stopwords=nltk.corpus.stopwords.words('english')

def	alpha_filter(w):
		#	pattern	to	match	a	word of	non-alphabetical	characters
				pattern	=	re.compile('^[^a-z]+$')
				if	(pattern.match(w)):
								return	True
				else:
								return	False

alphashortwords	=	[w	for	w	in	shortwords	if	not	alpha_filter(w)]
alphashortwords

print "------------------------------------------------------------------"
print "Top 50 words by frequency of Stephen Hawking(Non Fiction) book"
nfwords = [w.lower()for w in filetokens]
nfshortwords = nfwords
stoppednfshortwords=[w for w in nfshortwords if not w in stopwords]
shortdist=FreqDist(stoppednfshortwords)
shortdist.keys()
for word in shortdist.keys()[:50]:
    print word,shortdist[word]

					
print "################  Frequency Bigram ###########################"


bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(filewords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print " Lenth of Raw frequencies without applying filter  "
print len(scored)
print ""
print ""
for bscore in scored[:50]:
    print bscore



#Apply	the	non-alphabetical	and	stopword	filters,	noting	that	the	filters	substantially	reduce	
#the	numbers	of	bigrams	scored.
print "################  Frequency Bigram with 	non-alphabetical and "
print "             stopword filter ###########################"
stopWords   =	nltk.corpus.stopwords.words('english')
finder.apply_word_filter(alpha_filter)
finder.apply_word_filter(lambda w: w in stopWords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print " Lenth of Raw frequencies after applying filter  "
print len(scored)
print ""
print ""
print "##### Raw frequencies of top 20 bigrams by applying  Alphabetical "
print  "                             Filter and StopWords filter: #####"
for bscore in scored[:50]:
    print bscore
#This	scorer	shows	the bigrams	in	order	by	Pointwise	Mutual	Information.




print "####  Pointwise	Mutual	Information Scores ###########"
scored = finder.score_ngrams(bigram_measures.pmi)
print " Lenth of PMI score without applying frequency filter == "
print len(scored)
print ""
print ""
print " Top	20	bigrams	by	pointwise	mutual	information	scores:"
for bscore in scored[:50]:
    print bscore



#Without a  minimum	frequency filter,	the PMI	scores	tend	to	score	highly	on	bigrams	that	
#contain	unique	words,	which	includes	words	that	have	errors	in	tokenization.	 So	instead,	
#we	further	filter	the	bigrams	by	frequency	and	then	apply	the	PMI	scores.

print "#######  Applying frequency Filter on PMI #########"
finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.pmi)

print " Lenth of PMI score with applying frequency filter == " 
print len(scored)
print ""
print ""
print " PMI of top 20 frequencies by pplying Stopwords "
print " filter, alphabetical filter and frequency_count_filter:"
for bscore in scored[:50]: 
    print bscore
