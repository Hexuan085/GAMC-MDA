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

# def create_trigger_list(size, wmk_x, wmk_y):
#     result = []
#     wm_targets = np.loadtxt(os.path.join(wmk_x, wmk_y))
#     mid_file = 'pics'
#     for index in range(0, size):
#         image_path = f'{wmk_x}/{mid_file}/{index + 1}.jpg'
#         random_element = image_to_random_number(image_path)
#         random_element = str(random_element) + str(int(wm_targets[index]))
#         # random_element = random.randint(1, pow(2, 256))
#         print(int(wm_targets[index]))
#         print(random_element)
#         result.append(int(random_element))
#     return result

def create_trigger_list(size, path):
    result = []
    wm_targets = np.loadtxt(path)
    for index in range(0, size):
        result.append(int(wm_targets[index]))
    return result

# for i in range(n):
#     num_of_txs_in_block_temp.append((i + 1) * 20)
#     # total_utxo_set_size_for_merkle_tree = pow(2, 20)
#     total_utxo_set_size_for_accumulator = num_of_txs_in_block_temp[i] * 3
#     num_of_inputs_in_tx = 2
#     num_of_outputs_in_tx = 2
#     num_of_txs_in_block = num_of_txs_in_block_temp[i]
#
#     print(num_of_txs_in_block)
wmk = 'noise'
wmk_dict = {}
wmk_dict['wmk_type'] = wmk
num_of_txs_in_block = 1
num_of_inputs_in_tx = 30
num_of_outputs_in_tx = 1
total_utxo_set_size_for_accumulator = 100

print("--> initialize and fill up accumulator state")
n, A0, S = setup()  # n = p*q, A0 = secrets.randbelow(n), S空字典

# if total_utxo_set_size_for_accumulator < num_of_inputs_in_tx * num_of_txs_in_block:
#     print("please select larger total_utxo_set_size_for_accumulator.")

# print(total_utxo_set_size_for_accumulator)
path = f'/Users/hexuan/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/wm_png/{wmk}/wmk_mem.txt'
elements_for_accumulator = create_trigger_list(total_utxo_set_size_for_accumulator, path)
print('elements_for_accumulator', elements_for_accumulator)
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
wmk_dict['acc_batch_add_genesis_timing'] = acc_batch_add_genesis_timing
print("<--   Done.", acc_batch_add_genesis_timing[-1])

print("--> Generate Acc_Sign")
print(A_post_batch_add)
tik = time.time()
rsa_test = USE_RSA()
acc_Sign = rsa_test.rsaEncrypt(str(A_post_batch_add))
tok = time.time()
print("acc_Sign", acc_Sign)
# acc_de = rsa_test.rsaDecrypt(acc_Sign)
# print("acc_de", acc_de)

acc_sign_timing = tok - tik
wmk_dict['acc_sign_timing'] = acc_sign_timing
wmk_dict['acc'] = A_post_batch_add
wmk_dict['acc_Sign'] = acc_Sign
wmk_dict['inputs_for_accumulator'] = inputs_for_accumulator
print("<--   Done.", acc_sign_timing)

for i in range(num_of_txs_in_block):

    witnesses = create_all_membership_witnesses(A0, S, n)
    elements_list = list(S.keys())  # this specific order is important
    # print(witnesses)
    nonces_list = list(map(lambda x: S[x], elements_list))
    agg_wit, nipoe = aggregate_membership_witnesses(A_post_batch_add, witnesses, elements_list, nonces_list, n)

    is_valid = batch_verify_membership_with_NIPoE(nipoe[0], nipoe[1], agg_wit, elements_list, nonces_list, A_post_batch_add, n)
    print(is_valid)

print("--> prove single membership accumulator")
times = []
acc_single_mem_proofs = []
for i in range(num_of_txs_in_block):
    tik = time.time()
    inputs_list = []
    for j in range(num_of_inputs_in_tx):
        input_single = inputs_for_accumulator[num_of_inputs_in_tx * i + j]
        Is_Mem = prove_membership(A0, S, input_single, n)
        # print(Is_Mem)  #product of membership
        print(j)
        acc_single_mem_proofs.append(Is_Mem)
    tok = time.time()
    times.append(tok - tik)
sum_times = sum(times)
acc_single_prove_mem_timing.append(sum_times / len(times))  # average
wmk_dict['wit_single_mem_proofs'] = acc_single_mem_proofs
wmk_dict['acc_single_prove_mem_timing'] = sum_times
print("<--   Done. total:", sum_times, "; per tx:", acc_single_prove_mem_timing[-1])

print("--> prove batch membership accumulator")
times = []
acc_mem_proofs = []
for i in range(num_of_txs_in_block):
    tik = time.time()
    inputs_list = []
    for j in range(num_of_inputs_in_tx):
        inputs_list.append(inputs_for_accumulator[num_of_inputs_in_tx * i + j])
    Is_Mem = batch_prove_membership(A0, S, inputs_list, n)
    # print(Is_Mem)  #product of membership
    acc_mem_proofs.append(Is_Mem)
    tok = time.time()
    times.append(tok - tik)
sum_times = sum(times)
acc_batch_prove_mem_timing.append(sum_times / len(times))  # average
wmk_dict['wit_batch_mem_proofs'] = acc_mem_proofs
wmk_dict['acc_batch_prove_mem_timing'] = sum_times
print("<--   Done. total:", sum_times, "; per tx:", acc_batch_prove_mem_timing[-1])

print("--> prove membership accumulator with NI-PoE")
times = []
acc_mem_proofs_with_NIPoE = []
for i in range(num_of_txs_in_block):
    tik = time.time()
    inputs_list = []
    for j in range(num_of_inputs_in_tx):
        inputs_list.append(inputs_for_accumulator[num_of_inputs_in_tx * i + j])
    acc_mem_proofs_with_NIPoE.append(batch_prove_membership_with_NIPoE(A0, S, inputs_list, n, A_post_batch_add))
    tok = time.time()
    times.append(tok - tik)
sum_times = sum(times)
acc_batch_prove_mem_with_NIPoE_timing.append(sum_times / len(times))  # average
print("<--   Done. total:", sum_times, "; per tx:", acc_batch_prove_mem_with_NIPoE_timing[-1])

print("--> accumulator single verify membership")
tik = time.time()
for i in range(num_of_txs_in_block):
    for j in range(num_of_inputs_in_tx):
        print(verify_membership(A_post_batch_add, inputs_for_accumulator[j], S[nonces_for_accumulator[j]], acc_single_mem_proofs[j], n))
tok = time.time()
acc_single_verify_mem_per_block_timing.append(tok - tik)
wmk_dict['single_verify_timing'] = acc_single_verify_mem_per_block_timing
print("<--   Done. total (per block):", acc_single_verify_mem_per_block_timing[-1])

print("--> accumulator batch verify membership")
tik = time.time()
for i in range(num_of_txs_in_block):
    inputs_list = []
    nonces_list = []
    for j in range(num_of_inputs_in_tx):
        inputs_list.append(inputs_for_accumulator[num_of_inputs_in_tx * i + j])
        nonces_list.append(nonces_for_accumulator[num_of_inputs_in_tx * i + j])
    # TODO: nonces should be given by the proofs?
    nonces_list = list(map(lambda x: S[x], nonces_list))
    print(batch_verify_membership(A_post_batch_add, inputs_list, nonces_list, acc_mem_proofs[i], n))
    # assert batch_verify_membership(A_post_batch_add, inputs_list, nonces_list, acc_mem_proofs[i], n)
tok = time.time()
acc_batch_verify_mem_per_block_timing.append(tok - tik)
wmk_dict['batch_verify_timing'] = acc_batch_verify_mem_per_block_timing
print("<--   Done. total (per block):", acc_batch_verify_mem_per_block_timing[-1])

print("--> accumulator batch verify membership with NIPoE")
tik = time.time()
for i in range(num_of_txs_in_block):
    inputs_list = []
    for j in range(num_of_inputs_in_tx):
        inputs_list.append(inputs_for_accumulator[num_of_inputs_in_tx * i + j])
    # TODO: nonces should be given by the proofs?
    nonces_list = list(map(lambda x: S[x], inputs_list))
    # nonces_list[1] = 1
    print( batch_verify_membership_with_NIPoE(
        acc_mem_proofs_with_NIPoE[i][0],
        acc_mem_proofs_with_NIPoE[i][1],
        acc_mem_proofs_with_NIPoE[i][2],
        inputs_list,
        nonces_list,
        A_post_batch_add,
        n))
tok = time.time()
acc_batch_verify_mem_with_NIPoE_per_block_timing.append(tok - tik)
acc_batch_verify_mem_with_NIPoE_per_tx_timing.append((tok - tik) / num_of_txs_in_block)  # average
print("<--   Done. total (per block):", acc_batch_verify_mem_with_NIPoE_per_block_timing[-1], "; per tx:",
      acc_batch_verify_mem_with_NIPoE_per_tx_timing[-1])

print(wmk_dict)
# with open(f'~/Documents/Academic/RESEARCH/Projects/RSA-accumulator-master/TBW/wm_png/{wmk}/wmk_dict_{num_of_inputs_in_tx}.pkl', 'wb') as file:
#     pickle.dump(wmk_dict, file)
