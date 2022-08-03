import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree

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

def normalize(a):
    """Min-max normalize a list

    Input:  a: list

    Output: list of normalized values
    """

    amax = max(a)
    amin = min(a)
    
    anorm = [(x - amin) / (amax - amin) for x in a]

    return(anorm)


def balance(con, candidates, lexicon, columns=None):
    """Generate balanced subset of stimuli based on opposing condition, from list of all possible options.

    Input:  con: list of words
            candidates: candidate words
            lexicon: pandas df lexicon both con and candidates are pulled from (for normalizaton)
            columns: list of column names for lexicon (optional)

    Output: 
            array: sugjested condition
    """

    # Get columns
    if columns == None:
        columns = [x for x in lexicon[lexicon.columns[1:]] if pd.api.types.is_numeric_dtype(lexicon[x])]
        print(f"Columns being used for similarity calculation are: {', '.join([x for x in columns])}\n")

    else:
        for column in columns:
            if not pd.api.types.is_numeric_dtype(lexicon[column]):
                raise Exception("Input columns are not numeric.")


    # Normalize
    for column in columns:
        lexicon[column] = normalize(lexicon[column])
    
    # Get normalized condition and candidate items
    con_norm = lexicon.loc[con]
    candidates_norm = lexicon.loc[candidates]

    # Build KD Tree
    kdTree = KDTree(candidates_norm[columns])

    matched_items = []

    for word in con:
        word_data = con_norm.loc[word][columns]
        dd, ii = (kdTree.query(word_data, k=2))
        matched_items.append(lexicon.index.tolist()[ii[1]])

    matched = lexicon.loc[matched_items]

    print("\nSugjested Condition Items")
    print(matched, '\n')
    print("Condition Means")
    print(con_norm.describe().loc['mean'])

    print("\nSugjested Matched Means")
    print(matched.describe().loc['mean'])

    return(matched)


    

# Testing for stimuli balancing
if __name__ == "__main__":

    stimuli = ['canal', 'caper', 'cargo']
    lexicon = pd.read_csv('engstim.csv', index_col='word')
    candidates = lexicon.index.tolist()
    
    # filter out NA values
    lexicon.dropna()
    balance(stimuli, candidates, lexicon)#, columns=['bg_freq', 'word_freq', 'lev_avg'])