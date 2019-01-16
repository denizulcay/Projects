#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file contains helper functions that we used for BLASTP and stitching operations.
'''
import sys
import os
import numpy as np
import Levenshtein as lev


''' 
- reads in matches from blast outfile
- alignments[i] will contain matches of query_i in the training set
- the matches are an array of size num_matches x 5
- each match has form (query_idx_start, query_idx_end, match, match_idx_start, match_idx end)
'''
def get_alignments(match_file):
    
    alignments = {}
    query_alignments = None
    with open(match_file, 'r') as f:
        line = f.readline()
        while line:
            # matches for one query
            if 'Query=' in line:
                if query_alignments: # dump the alignments from the previous query
                    alignments[query] = np.array(query_alignments)
                query = int(line.split(' ')[-1]) # get the new query number
                query_alignments = [] # reset alignments
            # record a training sequence that has match(es) with the query
            elif '>' in line: # get matching sequence
                match = int(line.split(' ')[-1])
            # record the indices of the match
            elif 'Query ' in line: # get the alignment
                q = list(filter(None, line.split(' '))) # start idx is 1, end idx is -1
                f.readline(); line = f.readline()
                m = list(filter(None, line.split(' '))) # start idx is 1, end idx is -1
                align = [int(q[1])-1, int(q[-1]), match, int(m[1])-1, int(m[-1])]
                query_alignments.append(align)
            line = f.readline()
        alignments[query] = np.array(query_alignments) # dump final query alignments
    
    return alignments



'''
-Takes matches in the aa or q8 sequences
-If the levenshtein distance between them is in fact relatively small, 
    save the corresponding inter-residue distances 
'''
def get_stitches_tr(alignments, train_df, train_target_data, maxlen_seq, query_idx):
    
    stitches = np.zeros((1, maxlen_seq, maxlen_seq))
    seq_len = train_df['length'].values
    aa_tr, q8_tr = train_df[['sequence', 'q8']].values.T

    for i in range(len(train_df)):
        if i not in alignments.keys(): # matches not found for all sequences
            continue
        al = alignments[i] # load up the alignments for the current test ex
        good_als = []
        lev_dists = []
        for a in al:
            query_aa = aa_tr[query_idx][a[0]:a[1]]
            match_aa = aa_tr[a[2]][a[3]:a[4]]
            length = a[1] - a[0]
            d = lev.distance(query_aa, match_aa) # compare the q8 segments corresponding to this match
            if d > length/2.: # skip matches with large q8 distances; consider removing this; will test when GAN is trained
                continue
            good_als.append(a)
            lev_dists.append(d)

        # sort the good matches by the Levenshtein difference of q8
        lev_dists = np.array(lev_dists)
        ordered_a = np.argsort(-lev_dists)

        # that way best matches are pasted over the top
        for idx in ordered_a:
            a = good_als[idx]
            length = a[1] - a[0]
            stitches[i, a[0]:a[1], a[0]:a[1]] = train_target_data[a[2], a[3]:a[3]+length, a[3]:a[3]+length]
    
    return stitches


'''
-Takes matches in the aa or q8 sequences
-If the levenshtein distance between them is in fact relatively small, 
    save the corresponding inter-residue distances 
'''
def get_stitches_te(alignments, train_df, train_target_data, maxlen_seq, test_df):
    
    stitches = np.zeros((len(test_df), maxlen_seq, maxlen_seq))
    seq_len = test_df['length'].values
    aa_tr, q8_tr = train_df[['sequence', 'q8']].values.T
    aa_te, q8_te = test_df[['sequence', 'q8']].values.T

    for i in range(len(test_df)):
        if i not in alignments.keys(): # matches not found for all sequences
            continue
        al = alignments[i] # load up the alignments for the current test ex
        good_als = []
        lev_dists = []
        for a in al:
            query_aa = aa_te[i][a[0]:a[1]]
            match_aa = aa_tr[a[2]][a[3]:a[4]]
            length = a[1] - a[0]
            d = lev.distance(query_aa, match_aa) # compare the q8 segments corresponding to this match
            if d > length/2.: # skip matches with large q8 distances; consider removing this; will test when GAN is trained
                continue
            good_als.append(a)
            lev_dists.append(d)

        # sort the good matches by the Levenshtein difference of q8
        lev_dists = np.array(lev_dists)
        ordered_a = np.argsort(-lev_dists)

        # that way best matches are pasted over the top
        for idx in ordered_a:
            a = good_als[idx]
            length = a[1] - a[0]
            stitches[i, a[0]:a[1], a[0]:a[1]] = train_target_data[a[2], a[3]:a[3]+length, a[3]:a[3]+length]
    
    return stitches


'''
This function creates the databases and query lists needed to run BLAST with aa 
or q8 training sequences.
'''
def create_databases_tr(aa_tr, db_path):
    # creates aa/q8 databases for each training sequence
    for j in range(len(aa_tr)):
        with open(db_path + 'aa_db' + str(j) + '.fasta', 'w+') as out:
            for i in range(len(aa_tr)):
                if i != j:
                    out.write('>' + str(i) + '\n' + aa_tr[i] + '\n')
        # creates query list with only the training sequence            
        with open(db_path + 'aa_query' + str(j) + '.fasta', 'w+') as out:
            out.write('>' + str(0) + '\n' + aa_tr[j] + '\n')
            
    return True


'''
This function creates the database and query list needed to run BLAST with aa 
or q8 test sequences.
'''
def create_databases_te(aa_tr, aa_te, db_path):
    # creates aa/q8 database
    with open(db_path + 'aa_db.fasta', 'w+') as out:
        for i in range(len(aa_tr)):
            out.write('>' + str(i) + '\n' + aa_tr[i] + '\n')
    # and query list        
    with open(db_path + 'aa_query.fasta', 'w+') as out:
        for i in range(len(aa_te)):
            out.write('>' + str(i) + '\n' + aa_te[i] + '\n')

    return True


'''
This function is to call BLASTP for test sequences from a Python script. 
Note that it requires NCBI's BLAST library and is known to have compatibility 
issues with Python 2. It takes in the database path and the destination path 
and number of training samples and saves a match file in the destination 
directory.
'''
def blastp_tr(db_path, dst_path, samples_len):
    
    for j in range(samples_len):
        os.system("makeblastdb -in " + db_path + "aa_db"+str(j)+".fasta -dbtype prot -out aadb")
        os.system("blastp -db aadb -query " + db_path + "aa_query"+str(j)+".fasta -out " + dst_path + "aa_match"+str(j)+".txt")


'''
This function is to call BLASTP from a Python script. Note that it requires
NCBI's BLAST library and is known to have compatibility issues with Python 2.
It takes in the database path and the destination path and saves a match file
in the destination directory.
'''
def blastp_te(db_path, dst_path):
    
    os.system("makeblastdb -in " + db_path + "aa_db.fasta -dbtype prot -out aadb")
    os.system("blastp -db aadb -query " + db_path + "aa_query.fasta -out " + dst_path + "aa_match.txt")
    
    
'''
Input:
    img: source matrix, shape (m, n)
Output:
    mask: matrix of shape (m, n). For pixel [i, j] of mask, if img[i, j] > 0 
          mask[i, j] = 1. Else, (if img[i, j] = 0), mask[i, j] = 0.
'''
def binary_mask(img):
    
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask[img > 0] = 1
    
    return mask


'''
This function takes in the filepaths of a BLASTP output, predictions and 
destination and creates an npz file with the stitched predictions at the 
destination.
'''
def stitch_predictions(align_path, pred_path, dst_path):
    
    alignments_aa = get_alignments(align_path)
    stitches_aa = get_stitches_te(alignments_aa)
    pred = np.load(pred_path)
    
    predict_stitched = []
    
    for i in range(len(pred)):
        
        prediction = np.array(pred['arr_'+str(i)])
        stitches = stitches_aa[i]
        stitches_siz = np.array(stitches[:prediction.shape[0], :prediction.shape[1]])
        mask = binary_mask(stitches_siz)
        dst_mask = 1 - mask
        patched = np.multiply(prediction, dst_mask) + stitches_siz
        np.fill_diagonal(patched, 0)
        predict_stitched.append(patched)
    
    np.savez_compressed(dst_path, *predict_stitched)
    
