import random
import re
from unidecode import unidecode


sourceFile = "C:\\Users\\Jan Kus\\Shorts\\WordClassifier\\TrainingData\\raw\\jap.txt"
saveFile = "TrainingData\japanese.csv"

f = open(sourceFile, "r", encoding="utf8")
w = open(saveFile, "a", encoding="utf8")
patrn = r"['#\.]"
wordList = [l for l in f]

random.shuffle(wordList)
for word in wordList:
    word = unidecode(word)
    if word.islower() and not re.search(patrn, word) and len(word) > 5 and len(word) < 18:
        w.write(word)

w.close()
f.close()


# spanish: https://github.com/javierarce/palabras/blob/master/listado-general.txt ,english: https://gist.github.com/h3xx/1976236
