from nltk.corpus import brown
import math
import regex as re

#'firstWord',

# Global variables
TAGS = []
word_shows_tags_dict = {}
joined_tag_show_dict = {}
tag_show_dict = {}
pseudo_word_shows_tags_dict = {}

START_TAG_NUM = -1
STOP_TAG_NUM = -1

# Constants
PSEUDO_WORDS = ['twoDigitNum', 'fourDigitNum', 'othernum', 'containsDigitsAndAlpha', 'containsDigitAndDash',
                'containsDigitAndSlash', 'containsDigitAndComma', 'containsDigitAndPeriod',
                'allCaps', 'capPeriod', 'initCap', 'lowercase', 'other']
PSEUDO_WORD_PATTERN = ['[0-9]{2}', '[0-9]{4}', '[0-9]+', '[0-9a-zA-Z]+', '[0-9\-]+', '[\\\\\\0-9]+', '[0-9\,]+',
                       '[0-9\.]+', '[A-Z]+', '[A-Z]+\.', '[A-Z][a-z]+', '[a-z]+']
SMOOTH_PICK_TAG = 0
SMOOTH_CONST_PROB = 1
SMOOTH_ADD_ONE = 2
NO_PSEUDO_WORDS_DICTIONARY = 0
PSEUDO_WORDS_DICTIONARY = 1


def load_pseudo_word_show_dict():
    for pseudo_word in PSEUDO_WORDS:
        pseudo_word_shows_tags_dict[pseudo_word] = [0] * len(TAGS)
    for word in word_shows_tags_dict.keys():
        sum = 0
        for i in range(len(TAGS)):
            sum += word_shows_tags_dict[word][i]
        if sum < 5:
            pseudo = find_pseudo_word(word)
            for i in range(len(TAGS)):
                pseudo_word_shows_tags_dict[pseudo][i] += word_shows_tags_dict[word][i]
        else:
            pseudo_word_shows_tags_dict[word] = word_shows_tags_dict[word]


def find_pseudo_word(word):
    for i in range(len(PSEUDO_WORD_PATTERN)):
        pattern = re.compile(PSEUDO_WORD_PATTERN[i])
        if re.fullmatch(pattern, word):
            return PSEUDO_WORDS[i]
    return PSEUDO_WORDS[len(PSEUDO_WORDS)-1]


def get_training_n_test_sets():
    """"
    generate the training and test sets and returns them.
    """
    data = brown.tagged_sents(categories='news')
    a_training_set = data[0:round(0.9*len(data))]
    a_test_set = data[round(0.9*len(data)): len(data)]
    return a_training_set, a_test_set


def suffix_tag(tag):
    """"
    this function takes a tag and returns it stripped form everything after the '+' or '-'
    """
    for i in range(len(tag)):
        if tag[i] == '+' or tag[i] == '-':
            return tag[0:i]
    return tag


def load_tags(a_training_set):
    """
    this function loads all the tags that appear in the training set to the TAG_SET
    """
    global TAGS, START_TAG_NUM, STOP_TAG_NUM
    tag_set = set()
    for tagged_sent in a_training_set:
        for tagged_word in tagged_sent:
            # (word, tag) = tagged_word
            tag = tagged_word[1]
            tag = suffix_tag(tag)
            tag_set.add(tag)
    TAGS = sorted(tag_set)
    TAGS.append('START')
    START_TAG_NUM = len(TAGS)-1
    TAGS.append('STOP')
    STOP_TAG_NUM = len(TAGS)-1
    return


def load_dicts(training_set):
    """
    this function loads the values of the word_shows_tags_dict, the tag_show_dict and the joined_tag_show_dict
    from the training set.
    tag_show_dict[i] = how many times the tag number i appeared in the training set.
    word_shows_tags_dict[x][i] = how many times the word x appeared with the tag number i (the number refers to the
    index of the tag in the TAGS array)
    joined_tag_show_dict[i][j] = how many times the tag number j followed the tag number i in the training set
    (the number refers to the index of the tag in the TAGS array)
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict, START_TAG_NUM, STOP_TAG_NUM
    word_shows_tags_dict = {}
    joined_tag_show_dict = {}

    tag_show_dict = dict()
    tag_show_dict[START_TAG_NUM] = len(training_set)
    tag_show_dict[STOP_TAG_NUM] = len(training_set)

    for tagged_sent in training_set:
        last_tag_num = START_TAG_NUM

        for tagged_word in tagged_sent:

            # init
            word = tagged_word[0]
            tag = suffix_tag(tagged_word[1])
            tag_num = -1

            # finds the tag number for each word instance tag.
            for i in range(len(TAGS)):
                if tag == TAGS[i]:
                    tag_num = i
                    break

            # fill tag_show_dict (tags appearance counters).
            if tag_num in tag_show_dict:
                tag_show_dict[tag_num] += 1
            else:
                tag_show_dict[tag_num] = 1

            # fill the word_shows_tags_dict (each word tags appearance counters).
            if word in word_shows_tags_dict.keys():
                word_shows_tags_dict[word][tag_num] += 1
            else:
                word_shows_tags_dict[word] = [0] * len(TAGS)
                word_shows_tags_dict[word][tag_num] = 1

            # fill joined_tag_show_dict (each tag, next tag appearances counters).
            if last_tag_num in joined_tag_show_dict:
                joined_tag_show_dict[last_tag_num][tag_num] += 1
            else:
                joined_tag_show_dict[last_tag_num] = [0] * len(TAGS)
                joined_tag_show_dict[last_tag_num][tag_num] = 1

            # Update last_tag_num.
            last_tag_num = tag_num

        # fill joined_tag_show_dict for [last_word][STOP] sequence.
        tag_num = STOP_TAG_NUM
        if last_tag_num in joined_tag_show_dict:
            joined_tag_show_dict[last_tag_num][tag_num] += 1
        else:
            joined_tag_show_dict[last_tag_num] = [0] * len(TAGS)
            joined_tag_show_dict[last_tag_num][tag_num] = 1

    return


def compute_word_most_likely_tag(word):
    """
    this function compute mlt for given word.
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict

    if not(word in word_shows_tags_dict.keys()):
        return 'NN'

    max_tag_num = 0
    max_tag_shows = 0
    for i in range(len(TAGS)):
        if word_shows_tags_dict[word][i] > max_tag_shows:
            max_tag_shows = word_shows_tags_dict[word][i]
            max_tag_num = i
    return TAGS[max_tag_num]


def compute_error_rate_mlt(test_set):
    """
    this function calculate the error rate on test set according to MLT.
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict
    known_word_num = 0
    unknown_word_num = 0
    known_word_correct = 0
    unknown_word_correct = 0

    for tagged_sent in test_set:
        for tagged_word in tagged_sent:
            # compute the mlt for each word instance.
            word_mlt_tag = compute_word_most_likely_tag(tagged_word[0])

            # count how much word known or not, and how much correct.
            if tagged_word[0] in word_shows_tags_dict.keys():
                known_word_num += 1
                if word_mlt_tag == suffix_tag(tagged_word[1]):
                    known_word_correct += 1
            else:
                unknown_word_num += 1
                if word_mlt_tag == suffix_tag(tagged_word[1]):
                    unknown_word_correct += 1

    # compute error rate known words
    if known_word_num > 0:
        error_rate_mlt_known = 1 - (known_word_correct / known_word_num)
    else:
        error_rate_mlt_known = 0

    # compute error rate unknown words
    if unknown_word_num > 0:
        error_rate_mlt_unknown = 1 - (unknown_word_correct / unknown_word_num)
    else:
        error_rate_mlt_unknown = 0

    error_rate_mlt = 1-((known_word_correct + unknown_word_correct)/(known_word_num+unknown_word_num))
    return error_rate_mlt_known, error_rate_mlt_unknown, error_rate_mlt


def compute_transition_prob_hmm(tag1_num, tag2_num):
    """
    this function calculate the transition probability of y(i), knowing y(i-1).
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict
    # q(tag(i)|tag(i-1)) = count(tag(i), tag(i-1))/sum(y')count(tag1, y') = count(tag1,tag2)/count(tag1)
    # q(tag2|tag1) = count(tag1, tag2)/sum(y')count(tag1, y') = count(tag1,tag2)/count(tag1)
    count_tag1_tag2 = joined_tag_show_dict[tag1_num][tag2_num]
    count_tag1 = tag_show_dict[tag1_num]
    return count_tag1_tag2/count_tag1


def compute_emission_prob_hmm(tag_num, word, smooth, dict):
    """
     this function calculate the emission probability of a word, knowing its tag.
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict, pseudo_word_shows_tags_dict
    # e(word|tag) = count(word,tag)/sum(x')count(x',tag) = count(word,tag)/count(tag)

    count_tag = tag_show_dict[tag_num]

    #regular word dict
    if dict == NO_PSEUDO_WORDS_DICTIONARY:
        # Handling known words:
        if word in word_shows_tags_dict.keys():
            count_word_tag = word_shows_tags_dict[word][tag_num]

            if smooth == SMOOTH_ADD_ONE:
                count_word_tag += 1
                count_tag += len(word_shows_tags_dict.keys())

        # Add smoothing to known words when appear with new tags.
            else:
                if count_word_tag == 0:
                    return 1e-20

        # Handling Unknown words:
        else:
            count_word_tag = 0
            if smooth == SMOOTH_ADD_ONE:
                count_word_tag += 1
                count_tag += len(word_shows_tags_dict.keys())

            elif smooth == SMOOTH_PICK_TAG:
                # for unknown word we will choose TAGS[27] as the tag
                if tag_num == 58:  # choose 'NN' (58)
                    count_word_tag = 1
                else:
                    count_word_tag = 0
            elif smooth == SMOOTH_CONST_PROB:
                return 1e-20  # 0.00001

    # pseudo word dict
    else:
        if word in pseudo_word_shows_tags_dict:
            count_word_tag = pseudo_word_shows_tags_dict[word][tag_num]
        else:
            pseudo = find_pseudo_word(word)
            count_word_tag = pseudo_word_shows_tags_dict[pseudo][tag_num]
        if smooth == SMOOTH_ADD_ONE:
            count_word_tag += 1
            count_tag += len(pseudo_word_shows_tags_dict.keys())
    return count_word_tag/count_tag


def hmm_viterbi(sent, smooth, dict):
    """
    compute the viterbi algorithm
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict, START_TAG_NUM, STOP_TAG_NUM

    # Init parameters
    n = len(sent)
    m = len(TAGS)

    # matrix holding all tag number options for each word in sentence.
    S = list()
    # add S0= [*] for START
    S.append([START_TAG_NUM])
    # S += n cells each holding all tagging options
    for i in range(n):
        S.append(list(range(len(TAGS)-2)))
    S.append([STOP_TAG_NUM])

    # list of back-tracking pointers to tags numbers for constructing the max probability tagging sequence.
    # parent[i][j] = if the i word is tagged by j, its parent is tagged by this.
    parent = []
    for i in range(n+1):
        parent.append([-1]*m)

    # pi(k, u) = matrix holding probability for k-word to be tagged by tag u.
    pi = []
    # init pi[k] = m cells list (each for a tag). 0 < k < n+1
    for k in range(n+2):
        pi.append([0]*m)

    # init START, STOP probabilities.
    pi[0][START_TAG_NUM] = 1
    pi[n+1][STOP_TAG_NUM] = 1

    # go over words in sentence.
    for k in range(1, n+1):  # S(1) to S(n+1) as STOP
        # go over tags possibilities for this word.
        for j in S[k]:
            max_prob = -math.inf
            p = -1
            # go over tags possibilities for previous word.
            for l in S[k-1]:
                trans_prob = compute_transition_prob_hmm(l, j)
                emm_prob = compute_emission_prob_hmm(j, sent[k-1], smooth, dict)

                # calc log probability value
                if (trans_prob > 0) and (emm_prob > 0):
                    prob = pi[k-1][l] + math.log(trans_prob) + math.log(emm_prob)
                else:
                    prob = -math.inf

                # look for max prob according to previous value
                if prob > max_prob:
                    max_prob = prob
                    p = l

            # Update max_prob to pi and parent.
            pi[k][j] = max_prob
            parent[k - 1][j] = p

    # Add STOP effect
    for j in S[n+1]:
        max_prob = -math.inf
        p = -1
        for l in S[n]:
            trans_prob = compute_transition_prob_hmm(l, j)

            # calc log probability value
            if trans_prob > 0:
                prob = pi[n][l] + math.log(trans_prob)
            else:
                prob = -math.inf

            # look for max prob according to previous value
            if prob > max_prob:
                max_prob = prob
                p = l

        # Update parent according to prop_max.
        parent[n][j] = p

    # backtrack to find tagging for each word.
    last_word_tag_num = parent[n][STOP_TAG_NUM]
    res = [("", -1)]*n
    res[n-1] = (sent[n-1], TAGS[last_word_tag_num])
    curr_idx = last_word_tag_num
    for i in range(n-2, -1, -1):
        parent_idx = parent[i+1][curr_idx]
        res[i] = (sent[i], TAGS[parent_idx])
        curr_idx = parent_idx
    return res


def compute_error_rate_hmm(test_set, smooth, dict):
    """
    compute error rate in test set using hmm.
    """
    global TAGS, word_shows_tags_dict, joined_tag_show_dict, tag_show_dict
    known_word_num = 0
    unknown_word_num = 0
    known_word_correct = 0
    unknown_word_correct = 0

    if dict == NO_PSEUDO_WORDS_DICTIONARY:
        curr_word_show_dict = word_shows_tags_dict
    else:
        curr_word_show_dict = pseudo_word_shows_tags_dict

    for tagged_sent in test_set:
        sent = [tagged_sent[i][0] for i in range(len(tagged_sent))]
        hmm_tagged_sent = hmm_viterbi(sent, smooth, dict)
        for i in range(len(tagged_sent)):
            word = tagged_sent[i][0]
            word_real_tag = tagged_sent[i][1]
            word_hmm_tag = hmm_tagged_sent[i][1]
            if word in curr_word_show_dict.keys():
                known_word_num += 1
                if word_hmm_tag == suffix_tag(word_real_tag):
                    known_word_correct += 1
            else:
                unknown_word_num += 1
                if word_hmm_tag == suffix_tag(word_real_tag):
                    unknown_word_correct += 1

    error_rate_hmm = 0
    error_rate_hmm_known = 0
    error_rate_hmm_unknown = 0
    if known_word_num > 0:
        error_rate_hmm_known = 1-(known_word_correct/known_word_num)
    if unknown_word_num > 0:
        error_rate_hmm_unknown = 1-(unknown_word_correct/unknown_word_num)
    if (known_word_num+unknown_word_num) > 0:
        error_rate_hmm = 1-((known_word_correct + unknown_word_correct) / (known_word_num + unknown_word_num))
    return error_rate_hmm_known, error_rate_hmm_unknown, error_rate_hmm


def print_statistics(test_set):
    global word_shows_tags_dict
    know_only = []
    word_cnt = 0
    unknown_words_cnt = 0

    for sent in test_set:
        all_in = True
        for word in sent:
            word_cnt += 1
            if word[0] not in word_shows_tags_dict:
                unknown_words_cnt += 1
                all_in = False
        if all_in:
            know_only.append(sent)

    print("TEST SET STATISTICS:")
    print("unknown words amount: " + str(unknown_words_cnt))
    print("total words amount: " + str(word_cnt))
    print("ratio words (unknown from total): " + str(unknown_words_cnt/word_cnt))
    print("total sentences: " + str(len(test_set)))
    print("known words only sentences: " + str(len(know_only)))
    print("ratio sentences (known words only from total): " + str(len(know_only)/len(test_set)))
    print()
    return print_statistics


def print_dict(a):
    for i in a:
        print(i)
        print(a[i])


if __name__ == '__main__':

    training_set, test_set = get_training_n_test_sets()

    # training_set = [[("the", "NN"), ("dog", "NN"), ("the", "VB")],
    #                 [("dog", "VB"), ("the", "NN-ty"), ("dog", "ADJ"), ("dog", "VB")]]
    # test_set = [[("the", "NN"), ("dog", "ADJ"), ("ad", "ADJ")]]

    # training_set = [[("A", "H"), ("A", "H"), ("A", "L"), ("A", "L"), ("A", "L")],
    #                 [("C", "L"), ("C", "L"), ("G", "L"), ("G", "L"), ("C", "H")],
    #                 [("C", "H"), ("C", "H"), ("T", "L"), ("G", "H"), ("T", "L"), ("G", "H")],
    #                 [("G", "H"), ("T", "L"), ("T", "H"), ("T", "H")]]
    #
    # test_set = [[("G", "H"), ("C", "H"), ("C", "L"), ("A", "H"), ("A", "H")]]

    load_tags(training_set)
    load_dicts(training_set)
    load_pseudo_word_show_dict()


    # print_statistics(test_set)
    #
    # errorRateMLT = compute_error_rate_mlt(test_set)
    # print("MLT ERROR RATE:")
    # print("ERROR RATE KNOWN = " + str(errorRateMLT[0]))
    # print("ERROR RATE UNKNOWN = " + str(errorRateMLT[1]))
    # print("TOTAL ERROR RATE = " + str(errorRateMLT[2]))
    # print()
    #
    # errorRateHMM = compute_error_rate_hmm(test_set, SMOOTH_PICK_TAG, NO_PSEUDO_WORDS_DICTIONARY)
    # print("HMM ERROR RATE (picking 'NN' tag for unknown words):")
    # print("ERROR RATE KNOWN = " + str(errorRateHMM[0]))
    # print("ERROR RATE UNKNOWN = " + str(errorRateHMM[1]))
    # print("TOTAL ERROR RATE = " + str(errorRateHMM[2]))
    # print()
    #
    # errorRateHMM = compute_error_rate_hmm(test_set, SMOOTH_CONST_PROB, NO_PSEUDO_WORDS_DICTIONARY)
    # print("HMM ERROR RATE (return const small probability for unknown words):")
    # print("ERROR RATE KNOWN = " + str(errorRateHMM[0]))
    # print("ERROR RATE UNKNOWN = " + str(errorRateHMM[1]))
    # print("TOTAL ERROR RATE = " + str(errorRateHMM[2]))
    # print()
    #
    # errorRateHMM = compute_error_rate_hmm(test_set, SMOOTH_ADD_ONE, NO_PSEUDO_WORDS_DICTIONARY)
    # print("HMM add one ERROR RATE:")
    # print("ERROR RATE KNOWN = " + str(errorRateHMM[0]))
    # print("ERROR RATE UNKNOWN = " + str(errorRateHMM[1]))
    # print("TOTAL ERROR RATE = " + str(errorRateHMM[2]))

    errorRateHMM = compute_error_rate_hmm(test_set, SMOOTH_PICK_TAG, PSEUDO_WORDS_DICTIONARY)
    print("HMM ERROR RATE pseudo words:")
    print("ERROR RATE KNOWN = " + str(errorRateHMM[0]))
    print("ERROR RATE UNKNOWN = " + str(errorRateHMM[1]))
    print("TOTAL ERROR RATE = " + str(errorRateHMM[2]))
    print()

    errorRateHMM = compute_error_rate_hmm(test_set, SMOOTH_ADD_ONE, PSEUDO_WORDS_DICTIONARY)
    print("HMM add one ERROR RATE and pseudo:")
    print("ERROR RATE KNOWN = " + str(errorRateHMM[ 0 ]))
    print("ERROR RATE UNKNOWN = " + str(errorRateHMM[ 1 ]))
    print("TOTAL ERROR RATE = " + str(errorRateHMM[ 2 ]))
