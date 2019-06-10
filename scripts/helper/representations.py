"""Generates representations of bound pairs for use in models.
Some code reused from these files:
https://github.com/eliberis/parapred/blob/d13600a3d5697ebd5796576e1d6166aa1a519933/parapred/data_provider.py
https://github.com/eliberis/parapred/blob/d13600a3d5697ebd5796576e1d6166aa1a519933/parapred/structure_processor.py"""

aa_s = "CSTPAGNDEQHRKMILVFYWX" # X for unknown

NUM_FEATURES = len(aa_s) + 7 # one-hot + extra features

def one_to_number(res_str):
    return [aa_s.index(r) for r in res_str]

def aa_features():
    # Meiler's features
    prop1 = [[1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]]
    return np.array(prop1)


def seq_to_one_hot(res_seq_one):
    from keras.utils.np_utils import to_categorical
    ints = one_to_number(res_seq_one)
    feats = aa_features()[ints]
    onehot = to_categorical(ints, num_classes=len(aa_s))
    return np.concatenate((onehot, feats), axis=1)

def process_chains(ag_search, ab_h_chain, ab_l_chain, sequences, pdb, max_cdr_len):
    results = get_cdrs_and_contact_info(ag_search, ab_h_chain, ab_l_chain, sequences, pdb)

    # Convert to matrices
    # TODO: could simplify with keras.preprocessing.sequence.pad_sequences
    cdr_mats = []
    cont_mats = []
    cdr_masks = []

    if results is None:
        return None

    cdrs, contact, counters = results

    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Convert Residue entities to amino acid sequences
        cdr_chain = [r[0] for r in cdrs[cdr_name]]

        cdr_mat = seq_to_one_hot(cdr_chain)
        cdr_mat_pad = np.zeros((max_cdr_len, NUM_FEATURES))
        cdr_mat_pad[:cdr_mat.shape[0], :] = cdr_mat
        cdr_mats.append(cdr_mat_pad)

        cont_mat = np.array(contact[cdr_name], dtype=float)
        cont_mat_pad = np.zeros((max_cdr_len, 1))
        cont_mat_pad[:cont_mat.shape[0], 0] = cont_mat
        cont_mats.append(cont_mat_pad)

        cdr_mask = np.zeros((max_cdr_len, 1), dtype=int)
        cdr_mask[:len(cdr_chain), 0] = 1
        cdr_masks.append(cdr_mask)

    cdrs = np.stack(cdr_mats)
    lbls = np.stack(cont_mats)
    masks = np.stack(cdr_masks)

return cdrs, lbls, masks, counters