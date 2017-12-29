'''
Deniz Ulcay

Naive Bayes Spam Classifier

==============================================================================

c = 1 k = 1

Precision: 0.9516129032258065   Recall: 0.8805970149253731 
F-Score: 0.9147286821705426     Accuracy: 0.9802513464991023
==============================================================================
dev.txt

c = 1 k = 4

Precision: 0.9384615384615385   Recall: 0.9104477611940298 
F-Score: 0.9242424242424243     Accuracy: 0.9820466786355476
==============================================================================
test.txt

c = 1 k = 4

Precision: 0.9655172413793104   Recall: 0.8888888888888888 
F-Score: 0.9256198347107438     Accuracy: 0.9838420107719928


'''


import sys
import string
import math

def extract_words(text):
    
    msg = text.lower()
    msg.strip()
    
    for p in string.punctuation:
        msg = msg.replace(p, " ")
        
    words = msg.split()
    for word in words:
        word = word.strip()
        
    return words


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   


        self.collect_attribute_types(training_filename, 4)
        self.train(training_filename)          

    def collect_attribute_types(self, training_filename, k):
        
        data = open(training_filename, "r")
        stops = open("stopwords_mini.txt", "r")
        
        unique = {}
        exclude = []
        
        for line in stops.readlines():
            line = line.strip()
            exclude.append(line)
            
        for line in data.readlines():
            text = line.split('\t', 1)
            linel = extract_words(text[1])
            for word in linel:
                if (word not in unique) and (word not in exclude):
                    unique[word] = 1
                elif (word not in exclude):
                    count = unique[word]
                    unique[word] = count + 1
        
        words = []
        for key in unique:
            if (unique[key] >= k):
                words.append(key)
            
        data.close()
        self.attribute_types = set(words)
        

    def train(self, training_filename):
        
        training = open(training_filename, "r")
        prior = {}
        texts = []
        given = {}
        ham_score = 0
        ham_words = 0
        spam_score = 0
        spam_words = 0
        c = 1
        
        for line in training.readlines():

            layn = line.split('\t', 1)
            layn[1] = extract_words(layn[1])
            
            for word in layn[1]:
                if word not in self.attribute_types:
                    while word in layn[1]:
                        layn[1].remove(word)
            
            texts.append(layn)
            
            if(layn[0] == "ham"):
                ham_score += 1
                ham_words += len(layn[1])
            elif(layn[0] == "spam"):
                spam_score += 1
                spam_words += len(layn[1])

        prior["ham"] = ham_score / (ham_score + spam_score)
        prior["spam"] = spam_score / (ham_score + spam_score)

        for text in texts:
            for word in text[1]:
                
                if ((word,text[0]) not in given):
                    if text[0] == "ham":
                        given[(word,"ham")] = (1 + c) / (ham_words + (c * len(self.attribute_types)))
                        given[(word, "spam")] = c / (spam_words + (c * len(self.attribute_types)))
                    elif text[0] == "spam":
                        given[(word,"spam")] = (1 + c) / (spam_words + (c * len(self.attribute_types)))
                        given[(word,"ham")] = c / (ham_words + (c * len(self.attribute_types)))
                        
                else:
                    if text[0] == "ham":
                        count = given[(word,"ham")]
                        given[(word,"ham")] = count + 1 / (ham_words + (c * len(self.attribute_types)))
                    elif text[0] == "spam":
                        count = given[(word,"spam")]
                        given[(word,"spam")] = count + 1 / (spam_words + (c * len(self.attribute_types)))

        training.close()

        self.label_prior = prior # replace this
        self.word_given_label = given #replace this

    def predict(self, text):
        
        log_prob = {}
        given = self.word_given_label
        words = extract_words(text)
        joint_probability_ham = math.log(self.label_prior["ham"])
        joint_probability_spam = math.log(self.label_prior["spam"])
        
        for word in words:
            if (word,"ham") in given:
                joint_probability_ham += math.log(given[(word,"ham")])
            if (word,"spam") in given:
                joint_probability_spam += math.log(given[(word,"spam")])
        
        log_prob["ham"] = joint_probability_ham
        log_prob["spam"] = joint_probability_spam
        
        return log_prob #replace this


    def evaluate(self, test_filename):
        
        test = open(test_filename, "r")
        t_poz = 0
        f_poz = 0
        t_neg = 0
        f_neg = 0
        
        for line in test.readlines():
            
            ans, text = line.split('\t', 1)
            pred = self.predict(text)
            
            if (pred["spam"] >= pred["ham"]):
                if ans == "spam":
                    t_poz += 1
                elif ans == "ham":
                    f_poz += 1
                    
            else:
                if ans == "ham":
                    t_neg += 1
                elif ans == "spam":
                    f_neg +=1

        
        precision = t_poz / (t_poz + f_poz)
        recall = t_poz / (t_poz + f_neg)
        fscore = 2 * precision * recall / (precision + recall)
        accuracy = (t_poz + t_neg) / (t_poz + t_neg + f_poz + f_neg)
        
        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    
    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
