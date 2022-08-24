import numpy as np


def numbers2letters(self, sequences):  # Tranforming letters to numbers (1234 --> ATGC)
    """
    Converts numerical values to ATGC-format
    :param sequences: numerical DNA sequences to be converted
    :return: DNA sequences in ATGC format
    """
    if type(sequences) != np.ndarray:
        sequences = np.asarray(sequences)

    my_seq = ["" for x in range(len(sequences))]
    row = 0
    for j in range(len(sequences)):
        seq = sequences[j, :]
        assert (
            type(seq) != str
        ), "Function inputs must be a list of equal length strings"
        for i in range(len(sequences[0])):
            na = seq[i]
            if na == 1:
                my_seq[row] += "A"
            elif na == 2:
                my_seq[row] += "T"
            elif na == 3:
                my_seq[row] += "C"
            elif na == 4:
                my_seq[row] += "G"
        row += 1
    return my_seq
