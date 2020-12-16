import os
import sys
import json
import math
from tqdm import tqdm
import pandas as pd
from ltp import LTP
from collections import defaultdict
import numpy as np
import torch

def read_file(file, type='normal'):
    if type in ['stopword', 'juzhen', 'zjxdict', 'normal']:
        with open(file, 'r', encoding='utf-8')as f:
            data = f.readlines()
        return data


shot_matrix_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/shotjuzhen.txt'
stopwords_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/StopWordTable.txt'
frame_lexunits_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/dbo.lex_frame.csv'
frame_information_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/dbo.ch_en_name.csv'
match_words_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/dbo.matchwords.csv'
wordvec_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/dbo.frameVector.csv'
alert_words_file = '/mnt/nlp-lq/yujunshuai/code/QA/datasets/resource/alert_words.txt'

data_file = '/mnt/nlp-lq/yujunshuai/code/QA/data/train_data.json'

shot_matrix = read_file(shot_matrix_file)
stop_word = read_file(stopwords_file)
stop_word = [i.strip() for i in stop_word]

alert_words = read_file(alert_words_file)
alert_words = [i.strip() for i in alert_words]

frame_lexunits = pd.read_csv(frame_lexunits_file, encoding='utf-8')
frame_relations = pd.read_csv(frame_information_file, encoding='utf-8')
match_words = pd.read_csv(match_words_file, encoding='utf-8')
wordvec = read_file(wordvec_file)
wordvec = {item.split(',')[0].strip():list(map(float, item.split(',')[-1].strip()[1:].split("#")[0:100])) for item in wordvec}

ltp = LTP('base')


class Sample:
    class Sentence:
        def __init__(self):
            self.sent = ""
            self.artical_id = None
            self.paragraph_id = None
            self.paragraph_count = None
            self.sent_count = None
            self.sent_id = None
            self.tokens = None
            self.pos = None
            self.topic_score = 0.0
            self.perspective_score = 0.0
            self.frame_score = 0.0
            self.sen_score = 0.0
            self.pagerank = 0.0

    def __init__(self):
        self.tag = None
        self.question = None
        # 材料号/段落号/句子
        self.contents = defaultdict(dict)
        self.predict_answers = []
        self.answers = []

def construct_sample(data):

    def cut_word(sentence):
        tokens, hidden = ltp.seg([sentence])
        pos = ltp.pos(hidden)
        return [token for token in tokens[0] if token not in stop_word and token in wordvec], [pos[0][i] for i in range(len(pos[0])) if tokens[0][i] not in stop_word and tokens[0][i] in wordvec]

    sample = Sample()
    sample.question = Sample.Sentence()
    sample.question.sent = data['question']
    sample.question.tokens, sample.question.pos = cut_word(data['question'])
    sample.tag = data['tag']
    for material in data['infos']:
        sample.contents[material] = {}
        for i, paragraph in enumerate(data['infos'][material]):
            for j, item in enumerate(data['infos'][material][paragraph]):
                sentence, label = item[0], int(item[1])
                sent = Sample.Sentence()
                sent.sent = sentence
                sent.artical_id = int(material)
                sent.paragraph_id = int(paragraph)
                sent.paragraph_count = len(data['infos'][material])
                sent.sent_count = len(data['infos'][material][paragraph])
                sent.sent_id = j
                sent.tokens, sent.pos = cut_word(sentence)
                if len(sent.tokens) == 0:
                    continue
                assert len(sent.tokens) == len(
                    sent.pos), 'tokens length not equal to pos'
                if int(label) == 1:
                    sample.answers.append(sentence)
                if paragraph in sample.contents[material]:
                    sample.contents[material][paragraph].append(sent)
                else:
                    sample.contents[material][paragraph] = [sent]
    return sample


def predict(sample):
    def get_sentence_vector(sentence):
        sentence_vector = []
        for token in sentence.tokens:
            if token in wordvec:
                sentence_vector.append(wordvec[token])
        return sentence_vector

    def get_sentence_similar(vector_list_1, vector_list_2):
        score_1, score_2 = 0.0, 0.0
        if len(vector_list_1) == 0 or len(vector_list_2) == 0:
            return 0.0
        for vector_1 in vector_list_1:
            cos_score_list = [cosine_similarity(
                vector_1, vector_2) for vector_2 in vector_list_2]
            score_1 += max(cos_score_list)
        for vector_1 in vector_list_2:
            cos_score_list = [cosine_similarity(
                vector_1, vector_2) for vector_2 in vector_list_1]
            score_2 += max(cos_score_list)
        similar_score = score_1 / \
            len(vector_list_1) + score_2 / len(vector_list_2)
        return similar_score / 2

    def cosine_similarity(x, y, norm=True):
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)
        if x == y:
            return 1.0
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]]
                        for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1]))
                                * np.sqrt(sum(res[:, 2])))
        return 0.5 * cos + 0.5 if norm else cos

    def cal_sentences_topic_score(sample, threshold=6, score1_weight=0.7, score2_weight=0.3):
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for i, item in enumerate(sample.contents[material_id][pragraph_id]):
                    raw_score = 1.0
                    if item.paragraph_id == 0 or item.paragraph_id + 1 == item.paragraph_count:
                        raw_score *= 0.7
                    else:
                        raw_score *= 0.3
                    if item.sent_id == 0 or item.sent_id + 1 == item.sent_count:
                        raw_score *= 1
                    else:
                        raw_score *= (1 - math.log(item.sent_id) /
                                    math.log(item.sent_count))
                    sentence_vector = get_sentence_vector(item)
                    similar_score_list = []
                    for j, another in enumerate(sample.contents[material_id][pragraph_id]):
                        if j == i:
                            continue
                        another_vector = get_sentence_vector(another)
                        similar_score_list.append(get_sentence_similar(
                            sentence_vector, another_vector))
                    another_raw_score = sum(similar_score_list) / item.sent_count
                    item.topic_score = float(
                        raw_score * score1_weight + another_raw_score * score2_weight)

    def cal_sentence_perspective_score(sample, threshold=6, score1_weight=0.4, score2_weight=0.0, score3_weight=0.6):
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for i, item in enumerate(sample.contents[material_id][pragraph_id]):
                    score_1 = 0.0
                    if item.paragraph_id + 1 == item.paragraph_count:
                        if item.sent_id == 0 or item.sent_id + 1 == item.sent_count:
                            score_1 *= 1
                        else:
                            score_1 *= (1 - math.log(item.sent_id) /
                                        math.log(item.sent_count))
                        sentence_vector = get_sentence_vector(item)
                        similar_score_list = []
                        for j, another in enumerate(sample.contents[material_id][pragraph_id]):
                            if j == i:
                                continue
                            another_vector = get_sentence_vector(another)
                            similar_score_list.append(get_sentence_similar(
                                sentence_vector, another_vector))
                        score_2 = sum(similar_score_list) / item.sent_count
                        if any([token in alert_words for token in item.tokens]):
                            score_3 = 1
                        else:
                            score_3 = 0
                        item.perspective_score = score_1 * score1_weight + \
                            score_2 * score2_weight + score_3 * score3_weight
                    else:
                        item.perspective_score = 1.0 if any(
                            [token in alert_words for token in item.tokens]) else 0.0

    def cal_frame_score(sample):

        def get_lex_content(key, pos):
            lexunix = frame_lexunits['lexicalUnit'].values.tolist()
            framepos = frame_lexunits['pos_mark'].values.tolist()
            frame = frame_lexunits['frame_ename'].values.tolist()
            if key in lexunix:
                index = lexunix.index(key)
                if framepos[index] == pos:
                    return frame[index]
            return -1

        def getindex(data, key):
            data.columns.values.tolist()
            enname = data['en_name'].values.tolist()
            if key in enname:
                index = enname.index(key)
                return index
            return -1

        question_lex_frame = {}
        for word, pos in zip(sample.question.tokens, sample.question.pos):
            frame = get_lex_content(word, pos)
            if frame != -1:
                if word not in question_lex_frame:
                    question_lex_frame.setdefault(word, [])
                    question_lex_frame[word].append(frame)
                else:
                    question_lex_frame[word].append(frame)
        sen_lex_frame = {}
        sent_id = -1
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for i, item in enumerate(sample.contents[material_id][pragraph_id]):
                    sent_id += 1
                    for word, pos in zip(item.tokens, item.pos):
                        frame = get_lex_content(word, pos)
                        if frame != -1:
                            if str(sent_id) + " ".join(item.tokens) not in sen_lex_frame:
                                sen_lex_frame.setdefault(str(sent_id) + " ".join(item.tokens), [])
                                sen_lex_frame[str(sent_id) + " ".join(item.tokens)].append(frame)
                            else:
                                sen_lex_frame[str(sent_id) + " ".join(item.tokens)].append(frame)
                        else:
                            if str(sent_id) + " ".join(item.tokens) not in sen_lex_frame:
                                sen_lex_frame.setdefault(str(sent_id) + " ".join(item.tokens), [])
        sentence_distance = []
        for lex, lexframe in sen_lex_frame.items():
            match_count_1 = 0
            match_count_2 = 0
            for que, queframe in question_lex_frame.items():
                for q in queframe:
                    for l in lexframe:
                        if l == q:
                            match_count_1 += 1
                senindex = getindex(frame_relations, lexframe)
                questindex = getindex(frame_relations, queframe)
                if senindex != -1 and questindex != -1:
                    distence = shot_matrix[senindex][questindex]
                    if distence <= 2:
                        match_count_2 += 1
            sentence_distance.append(match_count_1 + match_count_2)

        min_match, max_match = 0xFFFF, 0
        for distance in sentence_distance:
            if distance < min_match:
                min_match = distance
            elif distance > max_match:
                max_match = distance
        if max_match != 0:
            sentence_distance = [i/(max_match-min_match)
                                 for i in sentence_distance]
        i = 0
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for item in sample.contents[material_id][pragraph_id]:
                    item.frame_score = sentence_distance[i]
                    i += 1

    def cal_sen_score(sample, threshold=0.7):
        key_word_list = match_words['key_word'].values.tolist()
        match_word_list = match_words['match_word'].values.tolist()

        def cal_match_score(sentence, question):
            expand_sentence_tokens = sentence.tokens + [key_word_list[match_word_list.index(
                token)] for token in sentence.tokens if token in match_word_list]
            expand_question_tokens = question.tokens + \
                [match_word_list[key_word_list.index(
                    token)] for token in sentence.tokens if token in key_word_list]
            return sum([token in expand_question_tokens for token in expand_sentence_tokens])

        question_vector = torch.tensor(get_sentence_vector(sample.question))
        question_vector = question_vector.unsqueeze(1)
        sentence_distance = []
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for i, item in enumerate(sample.contents[material_id][pragraph_id]):
                    distance = cal_match_score(item, sample.question)
                    sentence_vector = torch.tensor(get_sentence_vector(item))
                    question_vector_ = question_vector.expand(
                        question_vector.shape[0], sentence_vector.shape[0], question_vector.shape[2])
                    for q_vector in question_vector_:
                        cosine_score = torch.cosine_similarity(
                            sentence_vector, q_vector)
                    count = 0
                    for score in cosine_score:
                        if score * 0.6 > threshold:
                            count += 1
                    sentence_distance.append(distance + count)

        min_match, max_match = 0xFFFF, 0
        for distance in sentence_distance:
            if distance < min_match:
                min_match = distance
            elif distance > max_match:
                max_match = distance
        if max_match != 0:
            sentence_distance = [i/(max_match-min_match)
                                 for i in sentence_distance]
        i = 0
        
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for i, item in enumerate(sample.contents[material_id][pragraph_id]):
                    item.sen_score = sentence_distance[i]
                    i += 1

    def pagerank(sample, alpha = 0.6, threshold = 1e-4):
        lamada1, lamada2, lamada3, lamada4, beta1, beta2 = 0.4, 0.2, 0.2, 0.2, 1.0, 0.1
        total_sentences = [sample.contents[k][i][j]
                           for k in sample.contents for i in sample.contents[k] for j in range(len(sample.contents[k][i]))]
        weight = np.zeros((len(total_sentences)+1, len(total_sentences)+1), dtype=float)
        question_vector = get_sentence_vector(sample.question)
        sentence_sim_value = []
        
        for sentence in total_sentences:
            sentence_vector = get_sentence_vector(sentence)
            sentence_vector = torch.tensor(sentence_vector)
            similarity_matrix = []
            for i, question_word_vector in enumerate(question_vector):
                temp_question_vector = torch.tensor(question_word_vector).unsqueeze(0).expand(sentence_vector.shape[0], len(question_word_vector))
                similarity = torch.cosine_similarity(
                    sentence_vector, temp_question_vector)
                if sample.question.tokens[i] in sentence.tokens:
                    similarity[sentence.tokens.index(
                        sample.question.tokens[i])-1] = 1
                similarity_matrix.append(similarity.numpy())
            similarity_matrix = np.array(similarity_matrix)
            sentence_sim_value.append(np.average(
                np.max(similarity_matrix, 1)+np.average(np.max(similarity_matrix, 0)))/2)
        for j in range(len(total_sentences) + 1):
            for k in range(j + 1):
                if j == k:
                    weight[j][k] = 0
                elif k == 0:
                    weight[j][k] = beta1 * (lamada1 * total_sentences[j-1].sen_score + lamada2 * total_sentences[j-1].frame_score +
                                            lamada3 * total_sentences[j-1].topic_score + lamada4 * total_sentences[j-1].perspective_score)
                    weight[k][j] = weight[j][k]
                else:
                    weight[j][k] = beta2 * sentence_sim_value[j-1]
                    weight[k][j] = weight[j][k]
        cs = len(total_sentences) + 1
        Ranklist = []
        anslist = []
        old_importance = np.zeros((cs), dtype=float)
        weightp = np.zeros((cs), dtype=float)
        PR = np.zeros((cs), dtype=float)
        degreep = np.zeros((cs), dtype=float)

        for j in range(cs):
            n = cs
            for k in range(cs):
                if weight[j][k] == 0:
                    n -= 1
            degreep[j] = n
            if j == 0:
                old_importance[j] = 1.0
            else:
                old_importance[j] = 0.0
        while 1:
            for j in range(cs):
                sum_pro = np.zeros((cs), dtype=float)
                for k in range(cs):
                    weightp[k] = weight[j][k]
                    # wjk=degreep[0]
                    # for ww in range(1,k):
                    #     wjk+=degreep[ww]
                    sum_pro[j] += old_importance[k]*weightp[k]/n
                PR[j] = (1 - alpha) + alpha * sum_pro[j]
            diff = 0.0
            for j in range(cs):
                temp = abs(PR[j]-old_importance[j])
                if temp > diff:
                    diff = temp

                old_importance[j] = PR[j]
            if diff < threshold:
                for j in range(cs):
                    nodelist = {}

                    if j != 0:
                        nodelist['pagerank'] = old_importance[j]
                        nodelist['node'] = total_sentences[j-1].sent  # 句子

                    Ranklist.append(nodelist)
                for j in range(1, len(Ranklist)):
                    anslist.append(Ranklist[j])
                anslist = sorted(
                    anslist, key=lambda x: x['pagerank'], reverse=True)
                break
        result = []
        temp = []
        for an in anslist:
            if an['node'] not in temp:
                result.append(an)
                temp.append(an['node'])
                
        rank_dict = {}
        for i, node in enumerate(result):
            rank_dict[node['node']] = i
        
        for material_id in sample.contents:
            for pragraph_id in sample.contents[material_id]:
                for i, item in enumerate(sample.contents[material_id][pragraph_id]):
                    item.pagerank = rank_dict[item.sent]
                    
        return result
    
    cal_sentences_topic_score(sample)
    cal_sentence_perspective_score(sample)
    cal_frame_score(sample)
    cal_sen_score(sample)
    return pagerank(sample)


if __name__ == '__main__':
    all_precisions = []
    all_recalls = []
    with open(data_file) as f, open('train_res.json', 'w') as f1:
        for line in tqdm(f):
            if line.strip() == '':
                continue
            data = json.loads(line)
            sample = construct_sample(data)
            res = predict(sample)
            count = 0
            for r in res[:6]:
                if r['node'] in sample.answers:
                    count += 1
            all_precisions.append(count/6)
            all_recalls.append(count/len(sample.answers))
            print(f"precision: {all_precisions[-1]}, recall: {all_recalls[-1]}")
            print(f"precision: {sum(all_precisions)/len(all_precisions)}, recall: {sum(all_recalls)/len(all_recalls)}")
            data['pred'] = [item['node'] for item in res[:6]]
            # f1.write(json.dumps(data, ensure_ascii=False) + '\n')
            new_data = {}
            new_data['questiom'] = sample.question.sent
            sentences = [(sentence.sent, sentence.artical_id, len(sample.contents), sentence.paragraph_id, sentence.paragraph_count, sentence.sent_id, sentence.sent_count, sentence.topic_score, sentence.perspective_score, sentence.frame_score, sentence.sen_score, sentence.pagerank) for id in sample.contents for p in sample.contents[id] for sentence in sample.contents[id][p]]
            new_data['sentences'] = sentences
            new_data['labels'] = [1 if sent in sample.answers else 0 for sent in [item[0] for item in sentences]]
            f1.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            