import numpy as np
import xml.etree.ElementTree as ET

##### INPUT #####


def get_model(model_dir):
        vocab_pass = model_dir+"/vocab.all"
        vocab_dict = {}
        vocab_list = []
        vocab_list.append("")
        with open(vocab_pass, "r", encoding="utf_8_sig") as f:
                        data = f.read()
                        lines = data.split('\n')
                        v_id = 0
                        for line in lines:
                                if v_id == 0:
                                        v_id += 1
                                        continue
                                if line == "":
                                        break
                                vocab_dict[line] = v_id
                                vocab_list.append(line)
                                v_id +=1

        f_list_pass = model_dir+"/file-list"
        file_list = []
        with open(f_list_pass, "r") as f:
                        data = f.read()
                        file_list = data.split('\n')
                        
        return vocab_dict, vocab_list, file_list

##### FOR TF-IDF #####


def get_ngram_inv_f(model_dir, vocab_list):
        inv_f_pass = model_dir+"/inverted-file"
        with open(inv_f_pass, "r") as f:
                        data = f.read()
                        inv_f = data.split('\n')
        
        unigram_inv_f = []
        bigram_inv_f = []
        unigram_id_count_dict = {}
        bigram_id_count_dict = {}
        unigram_id_index_dict = {}
        bigram_id_index_dict = {}

        i=0
        prev_id1 = 0
        while i < len(inv_f):
                try:
                        id1, id2, q_cnt = map(int, inv_f[i].split(' '))
                        # UNI_GRAM
                        # if prev_id1 != id1 and id1 - prev_id1 > 1:
                        #         print("prev_id1, id1: {} {}".format(prev_id1, id1))
                        if id2 == -1:
                                unigram_inv_f.append(inv_f[i])
                                unigram_id_count_dict.setdefault(id1, q_cnt)
                                for j in range(i+1, i+q_cnt+1):
                                        unigram_inv_f.append(inv_f[j])
                                i += q_cnt + 1

                        # BI_GRAM
                        else:
                                bigram_inv_f.append(inv_f[i])
                                bigram_id_count_dict.setdefault((id1, id2), q_cnt)
                                for j in range(i+1, i+q_cnt+1):
                                        bigram_inv_f.append(inv_f[j])
                                i += q_cnt + 1
                        # prev_id1 = id1
                except:
                        # print("i: {}".format(i))
                        # print("j: {}".format(j))
                        # print("id1: {}".format(id1))
                        # print("len(inv_f): {}".format(len(inv_f)))
                        # print("len(unigram_inv_f)+len(bigram_inv_f): {}".format(len(unigram_inv_f)+len(bigram_inv_f)))
                        # print("len(vocab_list): {}".format(len(vocab_list)))
                        # print("len(unigram_id_count_dict): {}".format(len(unigram_id_count_dict)))
                        break
        # ### sort by frequency from high to low
        # unigram_id_count_dict = sorted(unigram_id_count_dict.items(), key=lambda x: x[1], reverse=True)
        # bigram_id_count_dict = sorted(bigram_id_count_dict.items(), key=lambda x: x[1], reverse=True)
       
        ####### reduce by low frequency words
        # scores_list = sorted(unigram_id_count_dict.items(), key=lambda x: x[1], reverse=True)
        # unigram_id_count_dict.clear()
        # try:
        #         unigram_id_count_dict.update(scores_list[:50000])
        # except:
        #         unigram_id_count_dict.update(scores_list)

        scores_list = sorted(bigram_id_count_dict.items(), key=lambda x: x[1], reverse=True)
        bigram_id_count_dict.clear()
        try:
                bigram_id_count_dict.update(scores_list[35000:45000])
        except:
                bigram_id_count_dict.update(scores_list)

        ## reduce by low frequency words
        
        # del_list = []

        # for key, value in unigram_id_count_dict.items():
        #         if value < 10:
        #                 del_list.append(key)
        # for key in del_list:
        #         unigram_id_count_dict.pop(key, None)
        
        # del_list = []
        # for key, value in bigram_id_count_dict.items():
        #         if value < 50:
        #                 del_list.append(key)
        # for key in del_list:
        #         bigram_id_count_dict.pop(key, None)

        ### index the position of term_id in the array
        for index, id in enumerate(unigram_id_count_dict.keys()):
                unigram_id_index_dict.setdefault(id,index)
        for index, id in enumerate(bigram_id_count_dict.keys()):
                bigram_id_index_dict.setdefault(id, index)
        return unigram_inv_f, bigram_inv_f, unigram_id_count_dict, bigram_id_count_dict, unigram_id_index_dict, bigram_id_index_dict


def get_dl_dict(NTCIR_dir, file_list):
        dl_list = []
        total_dl = 0
        for f_pass in file_list:
                if f_pass == "":
                        continue
                d = ""
                with open(NTCIR_dir+"/"+f_pass, "r") as f:
                        xml = f.read()
                        root = ET.fromstring(xml)
                        for text in root.iter('p'):
                                d += text.text
                        dl_list.append(len(d.replace('\n', '')))
                        total_dl += len(d.replace('\n', ''))
        # print(dl_list)
        # print(total_dl)
        return dl_list, total_dl


def get_okapi_idf(unigram_id_count_dict, bigram_id_count_dict, n_file):
        okapi_idf_uni = []
        okapi_idf_bi = []
        i = 0
        # ########################unigram
        # for q_cnt in unigram_id_count_dict.values():
        #         okapi_idf_uni.append(np.log((n_file - q_cnt + 0.5) / (q_cnt + 0.5)))
        # ########################
        for q_cnt in bigram_id_count_dict.values():
                okapi_idf_bi.append(np.log((n_file - q_cnt + 0.5) / (q_cnt + 0.5)))
        # print(len(okapi_idf_uni))
        # print(len(okapi_idf_bi))
        return okapi_idf_uni, okapi_idf_bi


def get_okapi_tf_idf(unigram_inv_f, bigram_inv_f, n_file, unigram_id_count_dict, bigram_id_count_dict, unigram_id_index_dict, bigram_id_index_dict, okapi_idf_uni, okapi_idf_bi, k, b, dl_list, avg_dl):
        okapi_tf_idf_uni = np.zeros((n_file, len(unigram_id_index_dict)))
        okapi_tf_idf_bi = np.zeros((n_file, len(bigram_id_index_dict)))
        okapi_tf_bi = np.zeros((n_file, len(bigram_id_index_dict)))
        # print("n_file: {}".format(n_file))
        # print("dl_list: {}".format(len(dl_list)))
        # print("len(unigram_id_index_dict: {}".format(len(unigram_id_index_dict)))
        # print("len(bigram_id_index_dict): {}".format(len(bigram_id_index_dict)))
        # print("len(unigram_inv_f): {}".format(len(unigram_inv_f)))
        # print("len(bigram_inv_f): {}".format(len(bigram_inv_f)))
        # print("len(okapi_idf_uni): {}".format(len(okapi_idf_uni)))
        # print("okapi_tf_idf_uni.shape(): {}".format(okapi_tf_idf_uni.shape()))
        
        ########################unigram
        # i = 0
        # while i < len(unigram_inv_f):
        #         # print("i : {}".format(i))
        #         # if i%10000 == 0:
        #                 # print("IN WHILE UNIGRAM  i = {}".format(i))
        #         # try:
        #         id1, id2, q_cnt = map(int, unigram_inv_f[i].split(' '))
        #         if id1 not in unigram_id_index_dict:
        #                 i += q_cnt + 1
        #                 continue
        #         for j in range(i+1, i+q_cnt+1):
        #                 # if j%50000 == 0:
        #                         # print("IN WHILE UNIGRAM  j = {}".format(j))
        #                 d_id, q_cnt_in_d = map(int, unigram_inv_f[j].split(' '))
        #                 # reference: https://ja.wikipedia.org/wiki/Okapi_BM25
        #                 # okapi_tf_idf_uni[d_id][unigram_id_index_dict[id1]] = (okapi_idf_uni[id1] * ((q_cnt_in_d * (k + 1)) / (q_cnt_in_d + k * (1 - b + b * dl_list[d_id] / avg_dl))))
        #                 # print("okapi_tf_idf_uni: {}".format(okapi_tf_idf_uni.shape))
        #                 # print("d_id: {}".format(d_id))
        #                 # print("unigram_id_index_dict[id1]: {}".format(unigram_id_index_dict[id1]))
                        
        #                 okapi_tf_idf_uni[d_id][unigram_id_index_dict[id1]] = okapi_idf_uni[unigram_id_index_dict[id1]]
        #                 okapi_tf_idf_uni[d_id][unigram_id_index_dict[id1]] *= np.round(((q_cnt_in_d * (k + 1)) / (q_cnt_in_d + k * (1 - b + b * dl_list[d_id] / avg_dl))),3)
        #         i += q_cnt + 1
        #         # except: 
        #         # print("unigram_inv_f[i]: {}".format(unigram_inv_f[i]))
        #         # print("unigram_inv_f[j]: {}".format(unigram_inv_f[j]))
        #         # print("i: {}".format(i))
        #         # print("j: {}".format(j))
        #         # print("d_id: {}".format(d_id))
        #         # print("unigram_id_index_dict(id1): {}".format(unigram_id_index_dict[id1]))
        #         # break
        ########################
        # print("bigram_id_index_dict: {}".format(bigram_id_index_dict))
        i = 0
        while i < len(bigram_inv_f):                
                # if i%10000 == 0:
                #         print("IN WHILE UNIGRAM  i = {}".format(i))
                # try:
                id1, id2, q_cnt = map(int, bigram_inv_f[i].split(' '))
                if (id1, id2) not in bigram_id_index_dict:
                        i += q_cnt + 1
                        continue
                for j in range(i+1, i+q_cnt+1):
                        # if j%50000 == 0:
                                # print("IN WHILE BIGRAM  j = {}".format(j))
                        d_id, q_cnt_in_d = map(int, bigram_inv_f[j].split(' '))
                        # print("id1: {}".format(id1))
                        # print("bigram_id_index_dict[id1]:{}".format(bigram_id_index_dict[id1]))
####################################################### NEED TO SPEED UP ############################################
#                         # reference: https://ja.wikipedia.org/wiki/Okapi_BM25
#                         okapi_tf_idf_bi[d_id][bigram_id_index_dict[(id1, id2)]] = okapi_idf_bi[bigram_id_index_dict[(id1, id2)]]
#                         okapi_tf_idf_bi[d_id][bigram_id_index_dict[(id1, id2)]] *= np.round(((q_cnt_in_d * (k + 1)) / (q_cnt_in_d + k * (1 - b + b * dl_list[d_id] / avg_dl))),3)

# #######################################       SETTING      #############
#                         # print(okapi_tf_idf_bi[d_id][bigram_id_index_dict[(id1, id2)]])
#                         # if okapi_tf_idf_bi[d_id][bigram_id_index_dict[(id1, id2)]] < 3:
#                         #         okapi_tf_idf_bi[d_id][bigram_id_index_dict[(id1, id2)]] = 0
#                         #############

#                 # print("id1: {}".format(id1))
#                 # print("bigram_id_index_dict[(id1, id2)]:{}".format(bigram_id_index_dict[(id1, id2)]))
#                 i += q_cnt + 1
###################################################### MODIFIED
                        okapi_tf_bi[d_id][bigram_id_index_dict[(
                            id1, id2)]] = q_cnt_in_d
                i += q_cnt + 1
        # print("1")
        # # okapi_tf_bi = sp.csr_matrix(okapi_tf_bi)
        # # print("2")
        # # okapi_idf_bi = okapi_idf_bi
        # print("3")
        # tmp1 = np.round(okapi_tf_bi * (k+1), 4)
        # print("4")
        # dl_list = np.array(dl_list)
        # ###
        # tmp2 = (k * (1 - b + b * dl_list / avg_dl))
        # print(tmp2.shape)
        # # okapi_tf_bi.data += np.array(tmp2[okapi_tf_bi.tocoo().row]).reshape(len(okapi_tf_bi.data))
        # okapi_tf_bi += np.repeat(tmp2.reshape(len(tmp2), -1),
        #                          okapi_tf_bi.shape[1], axis=1)
        # print(okapi_tf_bi.shape)
        # ###
        # # under = under.reshape(len(under), -1)
        # # under = np.repeat(under, okapi_tf_bi.shape[1], axis=1)
        # # # print(under.shape)
        # # # print(type(under))
        # # print("5")
        # # # print(okapi_tf_bi.shape)
        # # # print(type(okapi_tf_bi))
        # # okapi_tf_bi += under

        # # print("6")
        # # print("### upper")
        # # print(upper.shape)
        # # print(len(upper.data))
        # # print(type(upper))
        # # print("### okapi_tf_bi")
        # # print(okapi_tf_bi.shape)
        # # print(type(okapi_tf_bi))
        # # print(len(okapi_tf_bi.data))
        # # okapi_tf_bi.data = np.round(upper / okapi_tf_bi,4)
        # okapi_tf_bi.data = np.round(tmp1 / okapi_tf_bi, 4)
        okapi_tf_idf_bi = okapi_tf_bi * okapi_idf_bi
        # print("7")
        # print("### okapi_tf_bi")
        # print(okapi_tf_bi.shape)
        # print(type(okapi_tf_bi))
        # print("### okapi_idf_bi")
        # print(okapi_idf_bi.shape)
        # print(type(okapi_idf_bi))
        # okapi_tf_idf_bi = np.round(okapi_tf_bi * np.array(okapi_idf_bi).flatten(),4)

        # print(okapi_tf_idf_bi.shape)
        # print(type(okapi_tf_idf_bi))
        # print("8")

        return okapi_tf_idf_uni, okapi_tf_idf_bi







