
# lext
Lext (lexical toolkit): Miscellaneous tools for calculating lexical variables in psycholinguistic research

## Dependencies
Lext requires numpy, tqdm, and scipy. 
I've added the conda environment as a .yml file for easy setup.
Run `conda env create -f lext.yml` to set up the anaconda environment.
## Functions

### oup
Returns orthographic uniquness point for each word in lexicon.
**Example**:

    >>> oup(['cat', 'ca', 'cats', 'catalouge', 'catz'])
    [['ca', 'NA', -1], ['cat', 'NA', -1], ['catalouge', 'a', 3], ['cats', 's', 3], ['catz', 'z', 3]]

**Input**: list of strings
**Returns**: list of lists of format (word, OUP character, OUP index)

 
### lexical_overlap
Get average lexical overlap for each word lexicon.
**Example**:

    >>> test_list = ["aaa", "cc", "bb", "b", "a", "ab", "b$"]
    >>> lexical_overlap(test_list)
    [['a', 1], ['b', 1], ['ab', 2], ['b$', 2], ['bb', 3], ['cc', 1], ['aaa', 2]]

**Input**: list of strings
**Returns**: list of word, overlap sublists

### avg_bigram_frequency
Get average character bigram frequency based on word list.
Example:

    >>> avg_bigram_frequency(["aa", "aba", "abb"])
    [['aa', 1.0], ['aba', 1.5], ['abb', 1.5]]

**Input**: list of strings
**Returns**: list of word, avg frequency sublists

### balance
Generates a matched condition based on an input condition, based on multiple variables.
This function works by treating each word's lexical variables as coordinates, and then searches for the closest neighrbor in that multidemensinal space.
This function should create a matched set that has similar variation as the original set, but might not get as close of a match.

**Input:** 
- con: list of words, condition to be matched to
- candidates: candidate words
- lexicon: pandas df lexicon both con and candidates are pulled from (for normalizaton)
- columns (optional): list of column names for lexicon. Otherwise function considers all numeric columns
- verbose (optional), default True: Outputs stimuli stats
    
**Output**: lists of sugjested stimuli for matched condition

**Example**

Balance takes a *lexicon* input, which is a pandas dataframe such as the following:
| word  | oup | oup_char | bg_freq | word_freq | lev_avg |
|-------|-----|----------|---------|-----------|---------|
| aback | 4   | k        | 1032.75 | 59,7      | 7.47    |
| abaft | -1  | f        | 1222.5  | 2         | 7.05    |
| abbey | 4   | y        | 575     | 181       | 7.28    |

Using this lexicon, we can select the best words to balance the stimuli:

     >>> stimuli = ['canal', 'caper', 'cargo']
     >>> candidates  =  lexicon.index.tolist()
     >>> balance(stimuli, candidates, lexicon)
     
     Columns being used for similarity calculation are: bg_freq, word_freq, lev_avg
    
    
    Sugjested Condition Items
           oup oup_char   bg_freq  word_freq   lev_avg
    word                                              
    baton    3        o  0.578817   0.001482  0.274857
    mealy   -1      NaN  0.566994   0.000130  0.273624
    cords    4        s  0.306718   0.000049  0.457860 
    
    Condition Means
    oup          2.000000
    bg_freq      0.486541
    word_freq    0.002405
    lev_avg      0.339698
    Name: mean, dtype: float64
    
    Sugjested Matched Means
    oup          2.000000
    bg_freq      0.484176
    word_freq    0.000554
    lev_avg      0.335447
    Name: mean, dtype: float64

The output tells you the words selected to match, as well as shows the averages for each variable associated with both the list to match and the selected items.

### random_balance
Generate balanced subset of stimuli based on opposing condition, from list of all possible options.
Cycles through elminiating farthest outlier from con average each cycle.

**Input**:  
- con: list of words
- candidates: candidate words
- lexicon: pandas df lexicon both con and candidates are pulled from (for normalizaton)
- columns (optional): list of column names for lexicon. When set to None, calculation is based on all numeric columns passed.
- itr (optional): Number of balancing cycles

**Output**: list: sugjested condition

For an example, see balance.

