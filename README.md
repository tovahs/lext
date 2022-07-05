# lext
Lext (lexical toolkit): Miscellaneous tools for calculating lexical variables in psycholinguistic research

## Functions

### oup
Returns orthographic uniquness point for each word in lexicon.
**Example**:

    >>> oup(['cat', 'ca', 'cats', 'catalouge', 'catz'])
    [['ca', 'NA', -1], ['cat', 'NA', -1], ['catalouge', 'a', 3], ['cats', 's', 3], ['catz', 'z', 3]]

Input: list of words
Returns: list of lists of format (word, OUP character, OUP index)

 
### lexical_overlap
Get average lexical overlap for each word lexicon.
**Example**:

    >>> test_list = ["aaa", "cc", "bb", "b", "a", "ab", "b$"]
    >>> lexical_overlap(test_list)
    [['a', 1], ['b', 1], ['ab', 2], ['b$', 2], ['bb', 3], ['cc', 1], ['aaa', 2]]

### avg_bigram_frequency
Get average character bigram frequency based on word list.
Example:

    >>> avg_bigram_frequency(["aa", "aba", "abb"])
    [['aa', 1.0], ['aba', 1.5], ['abb', 1.5]]
