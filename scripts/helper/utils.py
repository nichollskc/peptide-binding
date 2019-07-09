"""Utils functions for rational design"""
import csv
import glob
import logging
import json
import os
import re


def get_map_float_to_str_pdb_ids():
    files = glob.glob("icMatrix/*_icMat.bmat")
    ids = [f.split("/")[1].split("_")[0] for f in files]
    exp_ids = [pdb_id for pdb_id in ids if re.match(r'\de\d\d', pdb_id)]
    exp_dict = dict([(str(float(pdb_id)), pdb_id) for pdb_id in exp_ids])
    return exp_dict


map_float_to_str_pdb_ids = get_map_float_to_str_pdb_ids()


def get_id_filename(pdb_id):
    """Given the pdb id, return the full filename for the IDs file."""
    return os.path.join("IDs/", pdb_id + "_ids.txt")


def get_matrix_filename(pdb_id):
    """Given the pdb id, return the full filename for the matrix file."""
    return os.path.join("icMatrix/", pdb_id + "_icMat.bmat")


def get_pdb_filename(pdb_id):
    """Given the pdb id, return the full filename for the PDB file."""
    return os.path.join("cleanPDBs2/", pdb_id + ".pdb")


def save_df_csv_quoted(data_frame, filename):
    """Saves a dataframe to a csv file, quoting everything to make it safer."""
    data_frame.to_csv(filename, header=True, index=False, quoting=csv.QUOTE_ALL)


def sanitise_pdb_id(pdb_id):
    """PDB ids such as 3e18 will have been read into pandas as numbers i.e. 3e+18
    To correct this, we can essentially just remove the +, along with any extra
    decimal places introduced."""

    # If the ID uses only numbers, the letter e, plus sign and dot then it might
    # be an ID that has been converted to a float
    sanitised = pdb_id
    if re.match(r'^[\de\+\.]*$', pdb_id):
        try:
            sanitised = map_float_to_str_pdb_ids[pdb_id]
        except KeyError:
            # This isn't an ID that has been converted from a float, so leave it alone
            pass
        if len(sanitised) != 4:
            match = re.match(r'(\d).*(e)\+.*(\d\d)', sanitised)
            if match:
                sanitised = ''.join(match.groups())
            else:
                logging.warning(f"PDB ID '{sanitised}' is longer than 4 characters, "
                                f"so suspect it has been converted to a float, but have "
                                f"failed to sanitise it.")
    return sanitised


def get_bound_pair_id_from_row(row):
    """Generate an ID for a bound pair that will be unique across all bound pairs."""
    cdr_pdb_id = sanitise_pdb_id(row['cdr_pdb_id'])
    cdr_indices = json.loads(row['cdr_bp_id_str'])
    cdr_length = cdr_indices[-1] - cdr_indices[0] + 1
    cdr_info = f"{cdr_pdb_id}-{cdr_indices[0]}-{cdr_length}"

    target_pdb_id = sanitise_pdb_id(row['target_pdb_id'])
    target_indices = json.loads(row['target_bp_id_str'])
    target_length = target_indices[-1] - target_indices[0] + 1
    target_info = f"{target_pdb_id}-{target_indices[0]}-{target_length}"

    if not row['binding_observed']:
        original_cdr_pdb_id = sanitise_pdb_id(row['original_cdr_pdb_id'])
        original_cdr_indices = json.loads(row['original_cdr_bp_id_str'])
        original_cdr_length = original_cdr_indices[-1] - original_cdr_indices[0] + 1
        original_cdr_info = f"{original_cdr_pdb_id}-" \
            f"{original_cdr_indices[0]}-{original_cdr_length}"
        full_id = f"{cdr_info}__{target_info}__{original_cdr_info}"
    else:
        full_id = f"{cdr_info}__{target_info}"

    return full_id


def get_row_from_bound_pair_id(bound_pairs_df, bound_pair_id):
    components = bound_pair_id.split("__")

    cdr_pdb_id, cdr_start_ind, cdr_length = components[0].split("-")
    cdr_indices_str = str(list(range(int(cdr_start_ind),
                                     int(cdr_start_ind) + int(cdr_length))))

    target_pdb_id, target_start_ind, target_length = components[1].split("-")
    target_indices_str = str(list(range(int(target_start_ind),
                                        int(target_start_ind) + int(target_length))))

    if len(components) == 3:
        original_cdr_pdb_id, original_cdr_start_ind, original_cdr_length = components[2].split("-")
        o_cdr_indices_str = str(list(range(int(original_cdr_start_ind),
                                           int(original_cdr_start_ind) + int(original_cdr_length))))
        rows = bound_pairs_df[(bound_pairs_df['cdr_pdb_id'] == cdr_pdb_id) &
                              (bound_pairs_df['cdr_bp_id_str'] == cdr_indices_str) &
                              (bound_pairs_df['target_pdb_id'] == target_pdb_id) &
                              (bound_pairs_df['target_bp_id_str'] == target_indices_str) &
                              (bound_pairs_df['original_cdr_pdb_id'] == original_cdr_pdb_id) &
                              (bound_pairs_df['original_cdr_bp_id_str'] == o_cdr_indices_str)]
    else:
        rows = bound_pairs_df[(bound_pairs_df['cdr_pdb_id'] == cdr_pdb_id) &
                              (bound_pairs_df['cdr_bp_id_str'] == cdr_indices_str) &
                              (bound_pairs_df['target_pdb_id'] == target_pdb_id) &
                              (bound_pairs_df['target_bp_id_str'] == target_indices_str)]

    assert len(rows) == 1,\
        f"Query using id string {bound_pair_id} should only have returned one row, " \
        f"instead {len(rows)} were returned"

    return rows


def get_dict_from_bound_pair_id(bound_pair_id):
    print(f"Interpreting bound pair id {bound_pair_id}")
    row_dict = {}
    components = bound_pair_id.split("__")

    cdr_pdb_id, cdr_start_ind, cdr_length = components[0].split("-")
    cdr_indices_str = str(list(range(int(cdr_start_ind),
                                     int(cdr_start_ind) + int(cdr_length))))

    target_pdb_id, target_start_ind, target_length = components[1].split("-")
    target_indices_str = str(list(range(int(target_start_ind),
                                        int(target_start_ind) + int(target_length))))

    if len(components) == 3:
        original_cdr_pdb_id, original_cdr_start_ind, original_cdr_length = components[2].split("-")
        o_cdr_indices_str = str(list(range(int(original_cdr_start_ind),
                                           int(original_cdr_start_ind) + int(original_cdr_length))))
        row_dict['cdr_pdb_id'] = cdr_pdb_id
        row_dict['cdr_bp_id_str'] = cdr_indices_str
        row_dict['target_pdb_id'] = target_pdb_id
        row_dict['target_bp_id_str'] = target_indices_str
        row_dict['original_cdr_pdb_id'] = original_cdr_pdb_id
        row_dict['original_cdr_bp_id_str'] = o_cdr_indices_str
        row_dict['binding_observed'] = 0
    else:
        row_dict['cdr_pdb_id'] = cdr_pdb_id
        row_dict['cdr_bp_id_str'] = cdr_indices_str
        row_dict['target_pdb_id'] = target_pdb_id
        row_dict['target_bp_id_str'] = target_indices_str
        row_dict['binding_observed'] = 1

    return row_dict
