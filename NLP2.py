from nltk.corpus import brown

data = brown.tagged_sents(categories='news')
len_data = len(data)
training_set = data[0:round(0.9*len_data)]
test_set = data[round(0.9*len_data): len_data]
"""
training_set = [[("the", "NN"), ("dog", "NN"), ("the", "VB")],[("dog", "VB"), ("the", "NN-ty"), ("dog", "ADJ"), ("dog", "VB")]]
test_set = [[("the", "NN"), ("dog", "ADJ")]]
"""
TAG_SET = set()
TAGS = []
word_shows_tags_dict = {}
joined_tag_show_dict = {}
tag_show_dict = {}

""""
this function takes a tag and returns it stripped form everithing after the '+' or '-'
"""
def suffix_tag(tag):
    for i in range(len(tag)):
        if (tag[ i ] == '+' or tag[ i ] == '-'):
            return tag[ 0:i ]
    return tag

"""
this function loads all the tags that appear in the training set to the TAG_SET
"""
def load_tags():
    for tagged_sent in training_set:
        for tagged_word in tagged_sent:
            tag = tagged_word[1]
            tag = suffix_tag(tag)
            TAG_SET.add(tag)

"""
this function loads the values of the word_shows_tag_dict, the tag_show_dict and the joined_tag_show_dict
from the training set
word_shows_tag_dict[x][i] = how many times the word x appeard with the tag number i (the number reffers to the
 index of the tag in the TAGS array)
joined_tag_show_dict[i][j] = how many times tha tag number j followed the tag number i in the training set
(the number reffers to the index of the tag in the TAGS array)
tag_show_dict[i] = how many times the tag number i appeared in the training set
"""
def load_dicts():
    word = ""
    tag = ""
    tag_show_dict['START'] = 0
    joined_tag_show_dict['START'] = [0] * len(TAGS)
    for tagged_sent in training_set:
        last_tag_num = None
        tag_show_dict['START']+=1
        for tagged_word in tagged_sent:
            word = tagged_word[ 0 ]
            tag = suffix_tag(tagged_word[1])
            tag_num = -1
            for i in range(len(TAGS)):
                if (tag == TAGS[ i ]):
                    tag_num = i
                    break
            if tag_num in tag_show_dict:
                tag_show_dict[tag_num] += 1
            else:
                tag_show_dict[tag_num] = 1
            if word in word_shows_tags_dict.keys():
                word_shows_tags_dict[ word ][ tag_num ] += 1
            else:
                word_shows_tags_dict[ word ] = [ 0 ] * len(TAGS)
                word_shows_tags_dict[ word ][ tag_num ] = 1
            if last_tag_num != None:
                if last_tag_num in joined_tag_show_dict:
                    joined_tag_show_dict[last_tag_num][tag_num] += 1
                else:
                    joined_tag_show_dict[last_tag_num] = [0] * len(TAGS)
                    joined_tag_show_dict[last_tag_num][tag_num] = 1
            else:
                joined_tag_show_dict['START'][tag_num] += 1
            last_tag_num = tag_num
    return


def compute_word_most_likely_tag(word):
    if (not(word in word_shows_tags_dict)):
        return 'NN'
    maxTagNum = 0
    maxTagShows = 0
    for i in range(len(TAGS)):
        if (word_shows_tags_dict[word][i] > maxTagShows):
            maxTagShows = word_shows_tags_dict[word][i]
            maxTagNum = i
    return TAGS[maxTagNum]

def compute_error_rate_MLT():
    known_word_num = 0
    unknown_word_num = 0
    known_word_correct = 0
    unknown_word_correct = 0
    for tagged_sent in test_set:
        for tagged_word in tagged_sent:
            word_MLT_tag = compute_word_most_likely_tag(tagged_word[0])
            if tagged_word[0] in word_shows_tags_dict.keys():
                known_word_num += 1
                if word_MLT_tag == suffix_tag(tagged_word[1]):
                    known_word_correct += 1
            else:
                unknown_word_num += 1
                if word_MLT_tag == suffix_tag(tagged_word[1]):
                    unknown_word_correct += 1
    if known_word_num > 0:
        errorRateMLT_known = 1-(known_word_correct/known_word_num)
    else:
        errorRateMLT_known = 0
    if unknown_word_num > 0:
        errorRateMLT_unknown = 1-(unknown_word_correct/unknown_word_num)
    else:
        errorRateMLT_unknown = 0
    errorRateMLT = 1-((known_word_correct + unknown_word_correct)/(known_word_num+unknown_word_num))
    return(errorRateMLT_known, errorRateMLT_unknown, errorRateMLT)


def compute_transition_prob_HMM(tag1_num, tag2_num):
    #q(tag2|tag1) = count(tag1, tag2)/sum(y')count(tag1, y') = count(tag1,tag2)/count(tag1)
    count_tag1_tag2 = joint_tag_count(tag1_num, tag2_num)
    count_tag1 = tag_count(tag1_num)
    return (count_tag1_tag2/count_tag1)

"""
for unknown word we will choose TAGS[0] as the tag
"""
def compute_emission_prob_HMM(tag_num, word):
    #e(word|tag) = count(word,tag)/sum(x')count(x',tag) = count(word,tag)/count(tag)
    if word in word_shows_tags_dict:
        count_word_tag = word_shows_tags_dict[word][tag_num]
    else:
        if(tag_num == 0):
            count_word_tag = 1
        else:
            count_word_tag = 0
    count_tag = tag_count(tag_num)
    return (count_word_tag/count_tag)

def tag_count(tag_num):
    return tag_show_dict[tag_num]

def joint_tag_count(tag1_num, tag2_num):
    return joined_tag_show_dict[tag1_num][tag2_num]

def HMM_Viterby(sent):
    n = len(sent)
    m = len(TAGS)
    S = []
    S.append(['*'])
    for i in range(n):
        S.append(TAGS)
    pi = []
    parent = []
    pi.append([0])
    for i in range(n):
        pi.append([0] * m)
        parent.append([-1]*m)
    pi[0][0] = 1
    for k in range(1,n+1):
        for j in range(m):
            max_prob = 0
            p = -1
            for l in range(len(S[k-1])):
                if(k != 1):
                    trans_prob = compute_transition_prob_HMM(l, j)
                else:
                    trans_prob = joined_tag_show_dict['START'][j]/tag_show_dict['START']
                emm_prob = compute_emission_prob_HMM(j, sent[k-1])
                if (pi[k-1][l]*trans_prob*emm_prob >= max_prob):
                    max_prob = pi[k-1][l]*trans_prob*emm_prob
                    p = l
            pi[k][j] = max_prob
            parent[k-1][j] = p
    max_prob = 0
    max_tag_idx = -1
    for tag_idx in range(m):
        if(pi[n][tag_idx] > max_prob):
            max_prob = pi[n][tag_idx]
            max_tag_idx = tag_idx
    res = [("", -1)]*n
    res[n-1] = (sent[n-1], max_tag_idx)
    curr_idx = max_tag_idx
    parent_idx = -1
    for i in range(n-2,-1,-1):
        parent_idx = parent[i+1][curr_idx]
        res[i] = (sent[i], parent_idx)
        curr_idx = parent_idx
    return res

def compute_error_rate_HMM():
    known_word_num = 0
    unknown_word_num = 0
    known_word_correct = 0
    unknown_word_correct = 0
    for tagged_sent in test_set:
        sent = [tagged_sent[i][0] for i in range(len(tagged_sent))]
        HMM_tagged_sent = HMM_Viterby(sent)
        for i in range(len(tagged_sent)):
            word = tagged_sent[i][0]
            word_real_tag = tagged_sent[i][1]
            word_HMM_tag = TAGS[HMM_tagged_sent[i][1]]
            if word in word_shows_tags_dict.keys():
                known_word_num += 1
                if word_HMM_tag == suffix_tag(word_real_tag):
                    known_word_correct += 1
            else:
                unknown_word_num += 1
                if word_HMM_tag == suffix_tag(word_real_tag):
                    unknown_word_correct += 1
    errorRateHMM = 0
    errorRateHMM_known = 0
    errorRateHMM_unknown = 0
    if known_word_num > 0:
        errorRateHMM_known = 1-(known_word_correct/known_word_num)
    if unknown_word_num > 0:
        errorRateHMM_unknown = 1-(unknown_word_correct/unknown_word_num)
    if (known_word_num+unknown_word_num) > 0:
        errorRateHMM = 1-((known_word_correct + unknown_word_correct)/(known_word_num+unknown_word_num))
    return(errorRateHMM_known, errorRateHMM_unknown, errorRateHMM)

if __name__ == '__main__':
    load_tags()
    TAGS = list(TAG_SET)

    load_dicts()
    errorRateMLT = compute_error_rate_MLT()
    print("MLT ERROR RATE:")
    print("ERROR RATE KNOWN = " + str(errorRateMLT[0]), "ERROR RATE UNKNOWN = " + str(errorRateMLT[1]),
          "TOTAL ERROR RATE = " + str(errorRateMLT[2]))

    errorRateHMM = compute_error_rate_HMM()
    print("HMM ERROR RATE:")
    print("ERROR RATE KNOWN = " + str(errorRateHMM[ 0 ]), "ERROR RATE UNKNOWN = " + str(errorRateHMM[ 1 ]),
          "TOTAL ERROR RATE = " + str(errorRateHMM[ 2 ]))