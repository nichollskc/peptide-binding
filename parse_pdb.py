from pymol import cmd, stored

import seaborn as sns
import matplotlib.pyplot as plt

import construct_database as con_dat

# Dictionaries to convert between one and three letter codes for amino acids
aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
aa1to3 = dict(zip(aa1, aa3))
aa3to1 = dict(zip(aa3, aa1))

#cmd.load("1atl.pdb")
# cmd.select("cdr", "chain A and resi 9-12")
# cmd.select("contact_atoms", "cdr around 3.5")
# cmd.select("contact_residues", "byres contact_atoms")
# cmd.save("contact_residues.pdb", "contact_residues")

# def find_cdr_targets(pdb_file, cdr_fragments_file):
#     cdr_fragments = con_dat.read_bound_pairs(cdr_fragments_file)
#
#     for cdr_fragment in cdr_fragment:
#         chain, start_ind, end_ind = cdr_fragment
#         cdr_select_string = "chain {} and resi {}-{}".format(chain,
#                                                              start_ind,
#                                                              end_ind)

def find_cdr_targets(chain, start_ind, end_ind):
    # Select the atoms in the CDR-like fragment
    cdr_select_string = "chain {} and resi {}-{}".format(chain, start_ind, end_ind)
    cmd.select("cdr", cdr_select_string)

    # Select all atoms within 3.5 Angstroms of the CDR-like atoms, excluding
    #   residues on either side of the fragment
    contacts_select_string = "(cdr around 3.5)"
    contacts_select_string += "and (not resi {}) and (not resi {})".format(start_ind - 1,
                                                                           end_ind + 1)
    cmd.select("contact_atoms", contacts_select_string)

    # Expand the selection to whole residues (byres)
    cmd.select("contact_residues", "byres contact_atoms")

    # Select only the alpha carbons of these residues and store in a list
    stored.list = []
    cmd.iterate("(contact_residues & n. ca)", "stored.list.append((resi, oneletter, chain))")

    # Split results into indices and residue codes, converting the three letter
    #   codes to one letter codes
    pdb_indices = [target[0] for target in stored.list]
    residues = [target[1] for target in stored.list]
    chains = [target[2] for target in stored.list]

    targets = {'pdb_indices': pdb_indices,
               'residues': residues,
               'chains': chains,
               'targets': stored.list}

    return targets

def find_cdr_targets_many(pdb_file, cdr_fragments_file):
    cdr_fragments = con_dat.read_bound_pairs(cdr_fragments_file)
    cmd.load(pdb_file)

    targets_combined = []
    targets_fragmented_combined = []

    full_lengths = []
    fragment_lengths = []

    for row in cdr_fragments.itertuples():
        cdr_pdb_indices = row.cdr_pdb_indices.split(",")
        start_ind = int(cdr_pdb_indices[0])
        end_ind = int(cdr_pdb_indices[-1])
        targets = find_cdr_targets(row.cdr_chain, start_ind, end_ind)

        print(row.cdr_residues)

        entry = {'cdr_residues': row.cdr_residues,
                 'cdr_chain': row.cdr_chain,
                 'cdr_pdb_indices': row.cdr_pdb_indices,
                 'target_residues': "".join(targets['residues']),
                 'target_length': len(targets['residues']),
                 'target_chain': targets['chains'][0],
                 'target_pdb_indices': ",".join(targets['pdb_indices'])}
        targets_combined.append(entry)

        full_lengths.append(len(targets['residues']))

        targets_fragmented = con_dat.find_contiguous_fragments(targets['targets'])

        for fragment in targets_fragmented:
            entry = {'cdr_residues': row.cdr_residues,
                     'cdr_chain': row.cdr_chain,
                     'cdr_pdb_indices': row.cdr_pdb_indices,
                     'target_residues': "".join([base[1] for base in fragment]),
                     'target_length': len(fragment),
                     'target_chain': fragment[0][2],
                     'target_pdb_indices': ",".join([base[0] for base in fragment])}
            targets_fragmented_combined.append(entry)
            if len(fragment) > 2:
                print(len(fragment))

            fragment_lengths.append(len(fragment))

    sns.distplot(full_lengths)
    plt.savefig("all_lengths.png")

    plt.clf()
    sns.distplot(fragment_lengths)
    plt.savefig("fragment_lengths.png")

    return targets_combined, targets_fragmented_combined

#find_cdr_targets_many("../example_files/1atl.pdb", "../example_files/1atl_bound_pairs_all.csv")
