import re
import numpy as np

def bigram_frequency(word_list):
    """ Get character bigram frequency from list of words.
    
    Inputs:
    word_list: list of strings.

    Returns:
    array: number of bigram instances per character combination of shape len(char) by len(char)
    chars: unique set of characters in lexicon. row/column ids for the arrays
    """

    chars = []

    for word in word_list:
        for char in word:
            chars.append(char)

    chars = list(set(chars))
    chars.sort()

    bg_array = np.zeros((len(chars) + 1, len(chars) + 1))

    for word in word_list:
        for x in range(len(word)):
            if x == 0:
                pass
            else:
                bg_array[chars.index(word[x]), chars.index(word[x - 1])] += 1
    
    return(bg_array, chars)

def avg_bigram_frequency(word_list):
    """Get average bigram frequency of a word.
    
    Inputs:
    word_list: list of strings

    Returns:
    list of lists of format [word, average character bigram frequency]
    """

    bg_array,chars = bigram_frequency(word_list)

    word_avgs = []

    for word in word_list:
        bg_freqs = []

        for x in range(len(word)):
            if x == 0:
                pass

            else:
                bg_freqs.append(bg_array[chars.index(word[x]), chars.index(word[x - 1])])
        #print(bg_freqs)

        if len(bg_freqs) != 0:
            word_avgs.append([word, sum(bg_freqs)/len(bg_freqs)])
        else:
            word_avgs.append([word, 0])
    
    return word_avgs

# Lexical Overlap
# Input is list of strings
def lexical_overlap(word_list):
    """Get average lexical overlap for each word lexicon.

    Parameters:
    word_list: list of strings

    Retunrs:

    """
    lex_overlap = []
    word_list = sorted(word_list, key=lambda x: (len(x), x))
    
    for x in range(len(word_list[-1]) + 1):
        length_word_list = [z for z in word_list if len(z) == x]

        word_string = " ".join(length_word_list)

        for word in length_word_list:

            # Remove special characters
            word = re.escape(word)

            # Itterate through characters in word

            word_overlap = 0
            for x in range(len(word)):
                if word[x] == "\\":
                    if x == 0:
                        pattern = "[^ ]" + word[x + 2:]
                        matches = re.findall(pattern, word_string)
                        word_overlap += len(matches)


                    else:
                        pattern = word[:x] + "[^ ]" + word[x + 2:]
                        matches = re.findall(pattern, word_string)
                        word_overlap += len(matches)

                elif word[x - 1] == "\\":
                    pass

                else:
                    # make regex pattern
                    if x == 0:
                        pattern = "[^ ]" + word[x + 1:]
                        matches = re.findall(pattern, word_string)
                        word_overlap += len(matches)

                    else:
                        pattern = word[:x] + "[^ ]" + word[x + 1:]
                        matches = re.findall(pattern, word_string)
                        word_overlap += len(matches)


            lex_overlap.append([re.sub(r'\\', "", word), word_overlap - 1])
    return(lex_overlap)

#test_list = ["aaa", "cc", "bb", "b", "a", "ab", "b$"]
#lexical_overlap(test_list)

# OUP
# Input corpus, return (word, oups) tupple list
def oup(word_list):
    """Returns orthographic uniquness point for each word in lexicon.

    Parameters:
    word_list: list of strings

    Returns:
    list of lists in format [word, OUP character, OUP]

    """
    oups = []
    word_list = sorted(word_list)

    with tqdm(total=len(word_list)) as pbar:
        for w_idx, word in enumerate(word_list):
            for c_idx, char in enumerate(word):
                #print(word[:c_idx+1])
                search_idx = np.searchsorted(word_list, word[:c_idx+1])

                if word == word_list[-1]:
                    if word[:c_idx +1] != word_list[w_idx - 1][:c_idx +1]:
                        oups.append([word, char, c_idx])
                        #print(char, search_idx)
                        break
                
                if word_list[search_idx+1].find(word[:(c_idx+1)]) < 0:
                    oups.append([word, char, c_idx])
                    #print(char, search_idx)
                    break
                
                if c_idx == len(word) - 1 :
                    oups.append([word, "NA", -1])

            pbar.update(1)
    
    return(oups)
    

#print(oup(['cat', 'ca', 'cats', 'catalouge', 'catz']))
