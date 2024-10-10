import time
from main import setup, batch_add, batch_prove_membership, batch_prove_membership_with_NIPoE, \
    batch_verify_membership, batch_verify_membership_with_NIPoE, batch_delete_using_membership_proofs,\
    aggregate_membership_witnesses,create_all_membership_witnesses, prove_membership, verify_membership
from helpfunctions import hash_to_prime, calculate_product
import csv
import os
import random
import numpy as np
import hashlib
from RSA_SIGN import USE_RSA
import pickle

# https://github.com/Tierion/pymerkletools
import merkletools
n = 1
num_of_txs_in_block_temp = []
acc_batch_add_genesis_timing = []
acc_batch_prove_mem_timing = []
acc_batch_prove_mem_with_NIPoE_timing = []
acc_batch_verify_mem_per_block_timing = []
acc_batch_verify_mem_per_tx_timing = []
acc_batch_verify_mem_with_NIPoE_per_block_timing = []
acc_batch_verify_mem_with_NIPoE_per_tx_timing = []
acc_single_prove_mem_timing = []
acc_single_verify_mem_per_block_timing = []
acc_single_verify_mem_per_tx_timing = []

def image_to_random_number(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
        hash_value = hashlib.sha256(data).hexdigest()
        random_number = int(hash_value, 16) 
        return random_number

def create_trigger_list(size, path):
    result = []
    wm_targets = np.loadtxt(path)
    for index in range(0, size):
        result.append(int(wm_targets[index]))
    return result

len_metric = {}
wmk_index = ['noise', 'unrelated', 'content', 'frontier_stitching', 'jia', 'blackmarks', 'deepmarks', 'deepsignwb']
for length in [30, 50, 100]:
    num_of_txs_in_block = 1
    num_of_inputs_in_tx = 3
    num_of_outputs_in_tx = 1
    wmk_dict = {}
    len_metric[('max', length)] = []
    len_metric[('min', length)] = []
    len_metric[('mean', length)] = []

    for wmk in wmk_index:
        wmk_dict = {}
        wmk_dict['wmk_type'] = wmk
        total_utxo_set_size_for_accumulator = length
        print("--> initialize and fill up accumulator state")
        n, A0, S = setup()  # n = p*q, A0 = secrets.randbelow(n), S空字典

        # if total_utxo_set_size_for_accumulator < num_of_inputs_in_tx * num_of_txs_in_block:
        #     print("please select larger total_utxo_set_size_for_accumulator.")

        # print(total_utxo_set_size_for_accumulator)

        path = f'/Users/hexuan/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/wm_png/{wmk}/wmk_mem.txt'
        elements_for_accumulator = create_trigger_list(total_utxo_set_size_for_accumulator, path)
        # print('elements_for_accumulator', elements_for_accumulator)
        inputs_for_accumulator = elements_for_accumulator[0:num_of_inputs_in_tx]  
        nonces_for_accumulator = elements_for_accumulator[0:num_of_inputs_in_tx]

        # print('inputs_for_accumulator', inputs_for_accumulator)
        # print('nonces_for_accumulator', nonces_for_accumulator)
        # outputs_for_accumulator = create_trigger_list(50, wmk_x, wmk_y)  
        tik = time.time()
        A_post_batch_add, proof = batch_add(A0, S, elements_for_accumulator, n) 
        # print('S')

        tok = time.time()

        acc_batch_add_genesis_timing.append(tok - tik)
        len_metric['acc_batch_add_genesis_timing'] = acc_batch_add_genesis_timing
        print("<--   Done.", acc_batch_add_genesis_timing[-1])

        sign_time = []
        for i in range(50):
            print("--> Generate Acc_Sign")
            # print(A_post_batch_add)
            print("length, wmk, i", length, wmk, i)
            tik = time.time()
            rsa_test = USE_RSA()
            print(A_post_batch_add)
            acc_Sign = rsa_test.rsaEncrypt(str(A_post_batch_add))
            tok = time.time()
            print("acc_Sign", acc_Sign)
            # acc_de = rsa_test.rsaDecrypt(acc_Sign)
            # print("acc_de", acc_de)

            acc_sign_timing = tok - tik
            sign_time.append(acc_sign_timing)
            print("<--   Done.", acc_sign_timing)

        len_metric[('max', length)].append(max(sign_time))
        len_metric[('min', length)].append(min(sign_time))
        len_metric[('mean', length)].append(np.mean(sign_time))

    print(len_metric)
    with open(f'~/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/wm_png/wmk_dict_{length}.pkl', 'wb') as file:
        pickle.dump(len_metric, file)
