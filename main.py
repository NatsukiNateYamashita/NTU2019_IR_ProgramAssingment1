import preprocess
import re
import pickle
import csv
import numpy as np
from MacOSFile import pickle_dump, pickle_load
import getopt
import sys
import scipy.sparse


def main(argv):
	Isrocchio = False
	try:
		options, args = getopt.getopt(argv, 'ri:o:m:d:', [])
	except getopt.GetoptError:
		print('Format: execute.sh [-r] -i query-file -o ranked-list -m model-dir -d NTCIR-dir')
	for opt, arg in options:
		if opt == '-r':
			Isrocchio = True
		elif opt == '-i':
			query_file = arg
		elif opt == '-o':
			ranked_list = arg
		elif opt == '-m':
			model_dir = arg
		elif opt == '-d':
			NTCIR_dir = arg
		else:
			pass
	return Isrocchio, query_file, ranked_list, model_dir, NTCIR_dir

Isrocchio, query_file, ranked_list, model_dir, NTCIR_dir = main(sys.argv[1:])

########## pING  ##########
########## GET TF and IDF ##########





# ##### TA docs #####
vocab_dict, vocab_list, file_list = preprocess.get_model(model_dir)
print("COMPLETED get_model")
# ### SAVE
# pickle_dump(vocab_dict, "vocab_dict")
# print("vocab_dict dumped")
# pickle_dump(vocab_list, "vocab_list")
# print("vocab_list dumped")
# pickle_dump(file_list, "file_list")
# print("file_list dumped")
## LOAD
# vocab_dict = pickle_load("vocab_dict")
# vocab_list = pickle_load("vocab_list")
# file_list = pickle_load("file_list")



#### BASIC INFO #####
unigram_inv_f, bigram_inv_f, unigram_id_count_dict, bigram_id_count_dict, unigram_id_index_dict, bigram_id_index_dict = preprocess.get_ngram_inv_f(model_dir,vocab_list)
print("COMPLETED get_ngram_inv_f")
### SAVE
# unigram_inv_f = scipy.sparse.csc_matrix(np.array(unigram_inv_f))
# # unigram_inv_f.todense()
# scipy.sparse.save_npz('unigram_inv_f.npz', unigram_inv_f)
# print("scipy saved")
# unigram_inv_f = scipy.sparse.load_npz('unigram_inv_f.npz')
# print("scipy loaded")
# bigram_inv_f = scipy.sparse.csc_matrix(np.array(bigram_inv_f))
# # bigram_inv_f.todense()
# scipy.sparse.save_npz('unigram_inv_f.npz', bigram_inv_f)
# print("scipy saved")
# bigram_inv_f = scipy.sparse.load_npz('unigram_inv_f.npz')
# print("scipy loaded")

# pickle_dump(unigram_inv_f, "unigram_inv_f")
# print("unigram_inv_f dumped")
# pickle_dump(bigram_inv_f, "bigram_inv_f")
# print("bigram_inv_f dumped")
# pickle_dump(unigram_id_count_dict, "unigram_id_count_dict")
# print("unigram_id_count_dict dumped")
# pickle_dump(bigram_id_count_dict, "bigram_id_count_dict")
# print("bigram_id_count_dict  dumped")
# pickle_dump(unigram_id_index_dict, "unigram_id_index_dict")
# print("unigram_id_index_dict dumped")
# pickle_dump(bigram_id_index_dict, "bigram_id_index_dict")
# print("bigram_id_index_dict  dumped")
## LOAD
# unigram_inv_f = pickle_load("unigram_inv_f")
# print("unigram_inv_f loaded")
# bigram_inv_f = pickle_load("bigram_inv_f")
# print("bigram_inv_f loaded")
# unigram_id_count_dict = pickle_load("unigram_id_count_dict")
# print("unigram_id_count_dict loaded")
# bigram_id_count_dict = pickle_load("bigram_id_count_dict")
# print("bigram_id_count_dict loaded")
# unigram_id_index_dict = pickle_load("unigram_id_index_dict")
# print("unigram_id_index_dict loaded")
# bigram_id_index_dict = pickle_load("bigram_id_index_dict")
# print("bigram_id_index_dict loaded")


# ### doc_length_normalization #####
dl_list, total_dl = preprocess.get_dl_dict(NTCIR_dir,file_list)
print("COMPLETED dl_list")
### SAVE
# pickle_dump(dl_list, "dl_list")
# print("dl_list dumped")
## LOAD
# dl_list = pickle_load("dl_list")
# print("dl_list loaded")

##### PARAMS #####
total_dl = 34864177
### total number of documents
# n_file = len(file_dict)
# print("n_file: {}".format(n_file))
n_file = 46972
### average document length
avg_dl = total_dl / n_file
### b for okapi bm25 DLN
b = 0.75
### k for okapi bm25 TF
k = 1.6

# # #### okapi_idf #####
okapi_idf_uni, okapi_idf_bi = preprocess.get_okapi_idf(unigram_id_count_dict, bigram_id_count_dict, n_file)
print("COMPLETED okapi_idf")
### SAVE
# pickle_dump(okapi_idf_uni, "okapi_idf_uni")
# print("okapi_idf_uni dumped") 
# pickle_dump(okapi_idf_bi, "okapi_idf_bi")
# print("okapi_idf_bi dumped")
## LOAD
# okapi_idf_uni = pickle_load("okapi_idf_uni")
# print("okapi_idf_uni loaded")
# okapi_idf_bi = pickle_load("okapi_idf_bi")
# print("okapi_idf_bi loaded")
##### SCIPY
# okapi_idf_uni = scipy.sparse.csc_matrix(okapi_idf_uni)
# okapi_idf_uni.todense()
# scipy.sparse.save_npz('okapi_idf_uni.npz', okapi_idf_uni)
# print("scipy saved")
# okapi_idf_uni = scipy.sparse.load_npz('okapi_idf_uni.npz')
# print("scipy loaded")

# okapi_idf_bi = scipy.sparse.csc_matrix(okapi_idf_bi)
# okapi_idf_bi.todense()
# scipy.sparse.save_npz('okapi_idf_bi.npz', okapi_idf_bi)
# print("scipy saved")
# okapi_idf_bi = scipy.sparse.load_npz('okapi_idf_bi.npz')
# print(okapi_idf_bi.shape)
# print("scipy loaded")

# #### okapi_tf_idf #####
okapi_tf_idf_uni, okapi_tf_idf_bi = preprocess.get_okapi_tf_idf(unigram_inv_f, bigram_inv_f, n_file, unigram_id_count_dict, bigram_id_count_dict, unigram_id_index_dict, bigram_id_index_dict, okapi_idf_uni, okapi_idf_bi, k, b, dl_list, avg_dl)
print("COMPLETED get_okapi_tf_idf")
# ### SAVE  w/ np
# np.save("okapi_tf_idf_uni", okapi_tf_idf_uni)
# print("okapi_tf_idf_uni dumped")
# np.save("okapi_tf_idf_bi", okapi_tf_idf_bi)
# print("okapi_tf_idf_bi dumped")
### SAVE w/ MacOSFile
# pickle_dump(okapi_tf_idf_uni, "okapi_tf_idf_uni.pkl")
# print("okapi_tf_idf_uni dumped")
# pickle_dump(okapi_tf_idf_bi, "okapi_tf_idf_bi.pkl")
# print("okapi_tf_idf_bi dumped")
# ### SAVE w/ pickle
# with open("okapi_tf_idf_uni.pkl", "wb") as f:
#         pickle.dump(okapi_tf_idf_uni, f)
# print("okapi_tf_idf_uni dumped")
# print("okapi_tf_idf_uni dumped")
# with open("okapi_tf_idf_bi.pkl", "wb") as f:
#         pickle.dump(okapi_tf_idf_bi, f)
# print("okapi_tf_idf_bi dumped")

### LOAD 
# okapi_tf_idf_uni = np.load(file="okapi_tf_idf_uni.npy")
# print("okapi_tf_idf_uni loaded")
# okapi_tf_idf_bi = np.load(file="okapi_tf_idf_bi.npy")
# print("okapi_tf_idf_bi loaded")
# okapi_tf_idf_uni = pickle_load("okapi_tf_idf_uni.pkl")
# okapi_tf_idf_bi = pickle_load("okapi_tf_idf_bi.pkl")

##### SCIPY
# okapi_tf_idf_uni = scipy.sparse.csc_matrix(okapi_tf_idf_uni)
# okapi_tf_idf_uni.todense()
# scipy.sparse.save_npz('okapi_tf_idf_uni.npz', okapi_tf_idf_uni)
# print("scipy saved")
# okapi_tf_idf_uni = scipy.sparse.load_npz('okapi_tf_idf_uni.npz')
# print("scipy loaded")

# okapi_tf_idf_bi = scipy.sparse.csc_matrix(okapi_tf_idf_bi)
# okapi_tf_idf_bi.todense()
# scipy.sparse.save_npz('okapi_tf_idf_bi.npz', okapi_tf_idf_bi)
# print("scipy saved")
# okapi_tf_idf_bi = scipy.sparse.load_npz('okapi_tf_idf_bi.npz')
# print("scipy loaded")



########## eING ##########
##########            ##########
#### STOPWORDS #####
stopword_list = extract.set_stopwords()
print("COMPLETED set_stopwords")
# ## SAVE
# pickle_dump(stopword_list, "stopword_list.pkl")
# print("stopword_list dumped")
### LOAD
# stopword_list = pickle_load("stopword_list.pkl")
# print("stopword_list loaded")



# #### QUERY pING #####
q_dict_uni, q_dict_bi, q_vocab_rowtf_idf_uni, q_vocab_rowtf_idf_bi = extract.query2vocab_id(query_file, vocab_dict, stopword_list, unigram_id_index_dict, bigram_id_index_dict, okapi_idf_uni, okapi_idf_bi)
print("COMPLETED query2vocab_id")
# # ### SAVE 
# pickle_dump(q_dict_uni, "q_dict_uni.pkl")
# print("q_dict_uni dumped")
# pickle_dump(q_dict_bi, "q_dict_bi.pkl")
# print("q_dict_bi dumped")

# # ## SAVE w/ np
# # np.save("q_vocab_rowtf_idf_uni", q_vocab_rowtf_idf_uni)
# # print("q_vocab_rowtf_idf_uni dumped")
# # np.save("q_vocab_rowtf_idf_bi", q_vocab_rowtf_idf_bi)
# # print("q_vocab_rowtf_idf_bi dumped")
# # ## SAVE w/ MacOSFile
# # pickle_dump(q_vocab_rowtf_idf_uni, "q_vocab_rowtf_idf_uni.pkl")
# # print("q_vocab_rowtf_idf_uni dumped")
# # pickle_dump(q_vocab_rowtf_idf_bi, "q_vocab_rowtf_idf_bi.pkl")
# # print("q_vocab_rowtf_idf_bi dumped")
# ### SAVE w/ picle
# with open("q_vocab_rowtf_idf_uni.pkl", "wb") as f:
#         pickle.dump(q_vocab_rowtf_idf_uni, f)
# print("q_vocab_rowtf_idf_uni dumped")
# with open("q_vocab_rowtf_idf_bi.pkl", "wb") as f:
#         pickle.dump(q_vocab_rowtf_idf_bi, f)
# print("q_vocab_rowtf_idf_bi dumped")

### LOAD 
# q_dict_uni = pickle_load("q_dict_uni.pkl")
# print("q_dict_uni loaded")
# q_dict_bi = pickle_load("q_dict_bi.pkl")
# print("q_dict_bi loaded")

# q_vocab_rowtf_idf_uni = np.load(file="q_vocab_rowtf_idf_uni.npy", allow_pickle=True)
# print("q_vocab_rowtf_idf_uni loaded")
# q_vocab_rowtf_idf_bi = np.load(file="q_vocab_rowtf_idf_bi.npy", allow_pickle=True)
# print("q_vocab_rowtf_idf_bi loaded")
# q_vocab_rowtf_idf_uni = pickle_load("q_vocab_rowtf_idf_uni.pkl")
# print("q_vocab_rowtf_idf_uni loaded")
# q_vocab_rowtf_idf_bi = pickle_load("q_vocab_rowtf_idf_bi.pkl")
# print("q_vocab_rowtf_idf_bi loaded")



# print("len(unigram_id_count_dict): {}".format(len(unigram_id_count_dict)))
# # get_okapi_idf
# print("len(bigram_id_count_dict): {}".format(len(bigram_id_count_dict)))
# print("len(okapi_idf_bi): {}".format(len(okapi_idf_bi)))
# # get_okapi_tf_idf
# print("len(bigram_id_index_dict): {}".format(len(bigram_id_index_dict)))
# print("okapi_tf_idf_bi.shape: {}".format(okapi_tf_idf_bi.shape))

### for experiment
# o_uni, o_scores_uni, o_bi, o_scores_bi, o_mix, o_scores_mix = e.get_ranking(okapi_tf_idf_uni, okapi_tf_idf_bi, q_vocab_rowtf_idf_uni, q_vocab_rowtf_idf_bi, file_list)

### for submit
output = extract.get_output(Isrocchio,okapi_tf_idf_uni, okapi_tf_idf_bi,q_vocab_rowtf_idf_uni, q_vocab_rowtf_idf_bi, file_list)
print("COMPLETED get_output")
with open(ranked_list+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)









