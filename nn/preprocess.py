# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]
    
    # Calculate number of sequences to sample from each class
    n_pos = len(pos_seqs)
    n_neg = len(neg_seqs)
    n_samples = min(n_pos, n_neg)
    
    # Sample sequences with replacement
    pos_samples = np.random.choice(pos_seqs, n_samples, replace=True)
    neg_samples = np.random.choice(neg_seqs, n_samples, replace=True)
    
    # Combine the sampled sequences and labels
    sampled_seqs = list(pos_samples) + list(neg_samples)
    sampled_labels = [True] * n_samples + [False] * n_samples
    
    # Shuffle the sequences and labels
    shuffle_idx = np.random.permutation(len(sampled_seqs))
    sampled_seqs = [sampled_seqs[i] for i in shuffle_idx]
    sampled_labels = [sampled_labels[i] for i in shuffle_idx]
    
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define the one-hot encoding dictionary
    encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

    # Initialize an empty list to store the encodings
    encodings = []

    # Iterate over each sequence in seq_arr
    for seq in seq_arr:
        # Initialize an empty list to store the one-hot encoding of this sequence
        encoding = []
        # Iterate over each nucleotide in the sequence
        for nt in seq:
            # Append the one-hot encoding of this nucleotide to the encoding list
            encoding += encoding_dict[nt]
        # Append the encoding to the list of encodings
        encodings.append(encoding)

    # Convert the list of encodings to a NumPy array and return it
    return np.array(encodings)