import numpy as np
import xml.etree.ElementTree as ET
import csv
from sklearn.metrics.pairwise import cosine_similarity
import re
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from operator import itemgetter
import heapq


def set_stopwords():
        stopwords = []
        with open('./stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                        if len(line) > 0:
                                stopwords.append(line.strip())
        return stopwords


def get_query(query_file):
        # f_pass = './wm-2020-vsm-model/queries/query-train.xml'
        # q_id = 0
        f_pass = query_file
        # q_id = 10
        with open(f_pass, "r") as f:
                xml = f.read()
                root = ET.fromstring(xml)

        query_dict = {}
        for child in root:
                number = child.find('number').text.replace("\n", "")
                title = child.find('title').text.replace("\n", "")
                question = child.find('question').text.replace("\n", "")
                narrative = child.find('narrative').text.replace("\n", "")
                concepts = child.find('concepts').text.replace("\n", "")

                topic = {
                    "number": number,
                    "title": title,
                    "question": question,
                    "narrative": narrative,
                    "concepts": concepts}
                q_id = number[15:]

                if number == "":
                        continue

                query_dict.setdefault(q_id, topic)
                # q_id += 1
        return query_dict


def query2vocab_id(query_file, vocab_dict, stopword, unigram_id_index_dict, bigram_id_index_dict, okapi_idf_uni, okapi_idf_bi):
        q_dict = get_query(query_file)
        ###################### UNI_GRAM
        q_vocab_rowtf_idf_uni = {}
        q_dict_uni = {}
        ###################### 
        q_vocab_rowtf_idf_bi = {}
        q_dict_bi = {}
        prev_v = ""
        # okapi_idf_bi = okapi_idf_bi.toarray().reshape(-1)
        for q_id in q_dict.keys():
                ###################### UNI_GRAM
                # count_v_in_q_uni = np.zeros(len(unigram_id_index_dict))
                ###################### 
                count_v_in_q_bi = np.zeros(len(bigram_id_index_dict))
                # print("-"*50)
                # print("q_id: {}".format(q_id))
                for q_type, query in q_dict[q_id].items():
                        ###################### UNI_GRAM
                        # vocab_id_list_uni = []
                        ###################### 
                        vocab_id_list_bi = []

######################################### SET UP
                        if q_type == "number" or q_type == "narrative":
                                continue
                        # print("{}: {}".format(q_type, query))
                        for v in query:
                                if v not in vocab_dict:
                                        continue
                                ###################### UNI_GRAM
                                # if v not in stopword and vocab_dict[v] in unigram_id_index_dict:
                                #         vocab_id_list_uni.append(vocab_dict[v])
                                #         count_v_in_q_uni[unigram_id_index_dict[vocab_dict[v]]] += 1
                                # # else:
                                #         # print("v is not in vocab_dict")
                                ######################
                                # BI_GRAM
                                
                                if v not in stopword and vocab_dict[v] in unigram_id_index_dict and prev_v != "":
                                        vocab_id_list_bi.append((vocab_dict[prev_v], vocab_dict[v]))
                                        if (vocab_dict[prev_v], vocab_dict[v]) in bigram_id_index_dict:
                                                count_v_in_q_bi[bigram_id_index_dict[(vocab_dict[prev_v], vocab_dict[v])]] += 1
                                        prev_v = v
                                elif v not in stopword and v in vocab_dict:
                                        prev_v = v
                                else:
                                        prev_v = ""
                                        # print("v is not in vocab_dict")
   
                        # print(vocab_id_list_uni)
                        # print(vocab_id_list_bi)

                        ###################### UNI_GRAM
                        # q_dict_uni.setdefault(q_id, {q_type: vocab_id_list_uni})
                        q_dict_bi. setdefault(q_id, {q_type: vocab_id_list_bi})

                # keep_list = []
                # for v_id in range(len(count_v_in_q_uni)):
                #         if count_v_in_q_uni[v_id] > 3:
                #                 keep_list.append(v_id)
                # count_v_in_q_uni = count_v_in_q_uni[keep_list]
                # keep_list = []
                # for v_id in range(len(count_v_in_q_bi)):
                #         if count_v_in_q_bi[v_id] > 3:
                #                 keep_list.append(v_id)
                # count_v_in_q_bi = count_v_in_q_bi[keep_list]

                # print("count_v_in_q_uni: {}".format(count_v_in_q_uni))
                # print("count_v_in_q_bi: {}".format(count_v_in_q_bi))
                # q_vocab_rowtf_uni.setdefault(q_id, count_v_in_q_uni)
                # q_vocab_rowtf_bi.setdefault(q_id, count_v_in_q_bi)
                ###################### UNI_GRAM
                # q_vocab_rowtf_idf_uni.setdefault(q_id, count_v_in_q_uni * okapi_idf_uni)
                # count_v_in_q_bi = sp.csr_matrix(count_v_in_q_bi)
                # okapi_idf_bi = sp.csr_matrix(okapi_idf_bi)
                # count_v_in_q_bi = np.array(count_v_in_q_bi)
                
                # count_v_in_q_bi = count_v_in_q_bi.reshape(50000, 1)
                # okapi_idf_bi = okapi_idf_bi.reshape(50000, 1)
                # okapi_idf_bi = np.ravel(okapi_idf_bi)
                # okapi_idf_bi = np.array(list(okapi_idf_bi))

                # print(count_v_in_q_bi.shape)
                # print(okapi_idf_bi.shape)
                # print(type(count_v_in_q_bi))
                # print(type(okapi_idf_bi))
                q_vocab_rowtf_idf_bi.setdefault(q_id, count_v_in_q_bi*okapi_idf_bi)
                # print(q_vocab_rowtf_idf_bi[q_id].shape)
        # print("len(_vocab_rowtf_idf_bi): {}".format(len(q_vocab_rowtf_idf_bi)))
        # print("len(count_v_in_q_bi): {}".format(len(count_v_in_q_bi)))
        # print("len(okapi_idf_bi): {}".format(len(okapi_idf_bi)))

        return q_dict_uni, q_dict_bi, q_vocab_rowtf_idf_uni, q_vocab_rowtf_idf_bi


def cos_sim(v1, v2):
        v1 = np.nan_to_num(v1, nan=np.finfo(np.float32).eps)
        v2 = np.nan_to_num(v2, nan=np.finfo(np.float32).eps)
        # np.seterr(all='raise')
        try:
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except:
                sim = 0
        return sim


def normarize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def normarize_1(v):
    return v / np.linalg.norm(v, keepdims=True)



################# OUTPUT #################
################# OUTPUT #################
def o_cos_sim_d_q(d_tfidf, q_tfidf, d_bool):

        scores = {}
        d_tfidf = np.round(d_tfidf, 3)
        # print("###cos")
        # ｑ_tfidf = np.array(q_tfidf.reshape(1,50000))
        # s = np.zeros((d_tfidf.shape[0],1))
        s = np.zeros(d_tfidf.shape[0])
############################################

        # print(d_tfidf.shape)
        # print(q_tfidf.shape)

        # d = normalize(d_tfidf)
        # q = normalize(q_tfidf)
        # d = sp.csr_matrix(d)
        # q = sp.csr_matrix(q)
        
        # s = np.round(d_tfidf*q_tfidf.T, 4)
        # s = s.toarray()

        # print(d.shape)
        # print(q.shape)
        # s = distance.cosine(d_tfidf,q_tfidf)
        # s = distance.cosine(d, q)
        # s = s.toarray()
        # for i in range(100):
        #         print(s[i])
        

############################################
        for index in d_bool:
                # print(d_id)
                d_tfidf[index] = np.array(d_tfidf[index])
                q_tfidf = np.array(d_tfidf[index])
                s[index] = cos_sim(d_tfidf[index], q_tfidf)
                # print("i: {}, score: {}".format(i, score))
                
                # scores.setdefault(index, s)
                scores.setdefault(index, s[index])


        # s[d_bool]= np.round(cosine_similarity(d_tfidf[d_bool], q_tfidf),4)
        # for index in d_bool:
        #         scores.setdefault(index, s[index])

        # scores = {}
        # for d_id in range(d_tfidf.shape[0]):
        #         # print(d_id)
        #         # d_tfidf[d_id] = np.array(d_tfidf[d_id])
        #         # q_tfidf = np.array(d_tfidf[d_id])
        #         # s = cos_sim(d_tfidf[d_id], q_tfidf)
        #         # print("i: {}, score: {}".format(i, score))
        #         scores.setdefault(d_id, s[d_id])
        # print("###sort")
        scores_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # s = list(s*-1)
        # heapq.heapify(s)
        # scores_list = sorted(scores.items(), key=itemgetter(1))[0:100]
        scores.clear()
        scores.update(scores_list)
        return scores


def o_retlieve(scores, thread, file_list):
        cnt = 0
        retlieved = []
        rocchio = []
        # while cnt > 99 or heapq.heappop(s)*(-1) < thread:
        #         rocchio.append(scores[s])
        #         retlieved.append(file_list[scores[s]])
        #         cnt +=1
        # for d_id, score in scores.items():
        #         if cnt > 99 or score < thread:
        #                 break
        #         if score > thread:
        #                 rocchio.append(d_id)
        #                 retlieved.append(file_list[d_id])
        #                 cnt += 1
        for d_id, score in scores.items():
                if cnt > 99 or score < thread:
                        break
                if score > thread:
                        rocchio.append(d_id)
                        retlieved.append(file_list[d_id])
                        cnt += 1
        # print("len(retlieved): {}, len(check_score): {}, len(roccho): {}".format(len(retlieved), len(check_score), len(rocchio)))
        return retlieved, rocchio


def o_get_scores_mix(uni, bi, r, thread, file_list, o_mix, o_scores_mix):

        for q_id, item in uni.items():
                # print("MIX::: q_id: {}".format(q_id))
                temp = {}
                for d_id, score in item.items():
                        # print("d_id: {}, score: {}".format(d_id, score))
                        tmp = r*score + (1-r)*bi[q_id][d_id]
                        temp.setdefault(d_id, tmp)
                # print("temp: {}".format(temp))
                retlieved, rocchio = o_retlieve(temp, thread, file_list)   
                for i in range(len(retlieved)):
                        retlieved[i] = retlieved[i][16:]  
                o_mix.append([q_id, " ".join(retlieved).lower()])
                # o_scores_mix.append([q_id+1, check_score])
                
        return o_mix


def get_output(Isrocchio,okapi_tf_idf_uni, okapi_tf_idf_bi, q_vocab_rowtf_idf_uni, q_vocab_rowtf_idf_bi, file_list):
        alpha = 1
        beta = 0.75
        gamma = 0.15
        thread_uni = 0.2
        afro_thread_uni = 0.2
        thread_bi = 0.15
        afro_thread_bi = 0.175
        thread_mix = 0.3
        r = 0.3
        # o_uni = [["query_id", "retrieved_docs"]]
        o_bi = [["query_id", "retrieved_docs"]]
        # o_mix = [["query_id", "retrieved_docs"]]
        # o_scores_uni = []
        # o_scores_bi = []
        o_scores_mix = []
        for_mix_uni = {}
        for_mix_bi = {}
        # okapi_tf_idf_bi = np.round(okapi_tf_idf_bi, 3)
        # okapi_tf_idf_bi = sp.csr_matrix(okapi_tf_idf_bi)
        # ##### UNI_GRAM
        # for q_id, q_rowtf_idf in q_vocab_rowtf_idf_uni.items():
        #         # q_rowtf_idf = np.array(q_rowtf_idf)
        #         # okapi_tf_idf_uni = mp.array(okapi_tf_idf_uni)
        #         # print("UNI::: q_id: {}".format(q_id))
        #         # print("cos")
        #         scores_uni = o_cos_sim_d_q(okapi_tf_idf_uni, q_rowtf_idf)
        #         # print("ret")
        #         retlieved, rocchio = o_retlieve(scores_uni, thread_uni, file_list)
        #         # print("n_rocchio")
        #         n_rocchio = list(set(range(len(okapi_tf_idf_uni)))-set(rocchio))
        #         # print("rocchio")
        #         new_q_tfidf = alpha * q_rowtf_idf + beta / len(rocchio) * np.sum(okapi_tf_idf_uni[rocchio], axis=0) - gamma / len(n_rocchio) * np.sum(okapi_tf_idf_uni[n_rocchio], axis=0)

        #         ### with rocchioed query
        #         scores_uni.clear()
        #         # print("cos")
        #         scores_uni = o_cos_sim_d_q(okapi_tf_idf_uni, new_q_tfidf)
        #         # retlieved, rocchio = o_retlieve(scores_uni, afro_thread_uni, file_list)
                
        #         # for i in range(len(retlieved)):
        #         #         retlieved[i] = retlieved[i][16:]  
        #         # o_uni.append([q_id+1, " ".join(retlieved).lower()])
        #         # o_scores_uni.append([q_id+1, check_score])
        #         for_mix_uni.setdefault(q_id+1, scores_uni)
        
        # # with open("o_uni.csv", "w") as f:
        # #         writer = csv.writer(f)
        # #         writer.writerows(o_uni)
        
        # ##### BI_GRAM
        for q_id, q_rowtf_idf in q_vocab_rowtf_idf_bi.items():
                # q_rowtf_idf = np.array(q_rowtf_idf)
                # okapi_tf_idf_bi = mp.array(okapi_tf_idf_bi)
                print("BI::: q_id: {}".format(q_id))
                ##### ROCCHIO
                q_bool = (q_rowtf_idf > 2)
                # print(q_rowtf_idf)
                d_bool = np.any(okapi_tf_idf_bi[:,q_bool] > 0, axis=1)
                d_bool = [i for i, x in enumerate(d_bool) if x == True]
                print("original: {}  booled: {}".format(okapi_tf_idf_bi.shape, okapi_tf_idf_bi[d_bool].shape))
                # q_rowtf_idf = sp.csr_matrix(np.array(q_rowtf_idf.reshape(1, 50000)))
                scores_bi= o_cos_sim_d_q(okapi_tf_idf_bi, q_rowtf_idf, d_bool)
                retlieved, rocchio = o_retlieve(scores_bi, thread_bi, file_list)
                
                if Isrocchio is True:
                        n_rocchio = list(set(range(okapi_tf_idf_bi.shape[0]))-set(rocchio))
                        try:
                                new_q_tfidf = alpha * q_rowtf_idf + beta / len(rocchio) * np.sum(okapi_tf_idf_bi[rocchio], axis=0) - gamma / len(n_rocchio) * np.sum(okapi_tf_idf_bi[n_rocchio], axis=0)
                        except:
                                new_q_tfidf = alpha * q_rowtf_idf 
                        ### with rocchioed query
                        scores_bi.clear()
                        # q_bool = (q_rowtf_idf > 3)
                        # d_bool = np.any(okapi_tf_idf_bi[:,q_bool]>0, axis=1)
                        # d_bool = [i for i, x in enumerate(d_bool) if x == True]
                        scores_bi = o_cos_sim_d_q(okapi_tf_idf_bi, new_q_tfidf, d_bool)
                        retlieved, rocchio = o_retlieve(scores_bi, afro_thread_bi, file_list)
                
                for i in range(len(retlieved)):
                        retlieved[i] = retlieved[i][16:]  
                o_bi.append([q_id, " ".join(retlieved).lower()])

                # o_scores_bi.append([q_id+1, check_score])
                for_mix_bi.setdefault(q_id, scores_bi)
        with open("o_bi.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(o_bi)
        ### MIX
        # o_mix = o_get_scores_mix(for_mix_uni, for_mix_bi, r, thread_mix, file_list, o_mix, o_scores_mix)
        # with open("o_mix_{}.csv".format(r), "w") as f:
        #         writer = csv.writer(f)
        #         writer.writerows(o_mix)
        # with open("o_scores_mix_{}".format(r), "w") as f:
        #         writer = csv.writer(f)
        #         writer.writerows(o_scores_mix)
        return o_bi
