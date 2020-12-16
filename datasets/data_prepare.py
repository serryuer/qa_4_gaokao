import os, sys, json, re
from ltp import LTP

ltp = LTP('base')

root_dir = '/mnt/nlp-lq/yujunshuai/code/QA/data/raw/train'
save_path = '/mnt/nlp-lq/yujunshuai/code/QA/data/train_data.json'

folders = {folder : os.path.join(root_dir, folder) for folder in os.listdir(root_dir)}
folders = {tag: [os.path.join(folder, item) for item in os.listdir(folder)] for tag, folder in folders.items()}

def construct_sample(question_file, resource_file, answer_file):
    with open(question_file) as f1, open(resource_file) as f2, open(answer_file) as f3:
        question = f1.readlines()
        resource = f2.readlines()
        answers = f3.readlines()
    answers = [item.strip() for item in answers if item.strip() != '']
    answers = [item if item[-1] not in ['。', '！', '？' '!', '?'] else item[:-1] for item in answers if item.strip() != '']
    assert len(question) == 1, 'question text larger than 2 lines'
    question = question[0].strip()
    paragraphs = [item.strip() for item in resource if item.strip() != '']
    material_count = 0
    paragraph_count = 0
    infos = {}
    infos[material_count] = {}
    for i, paragraph in enumerate(paragraphs):
        if paragraph.startswith('材料') and len(paragraph) == 3:
            if i != 0:
                material_count += 1
                infos[material_count] = {}
                paragraph_count = 0
            continue
        infos[material_count][paragraph_count] = [[item.strip(), 1 if item.strip() in answers else 0] for item in re.split(r'[。！？!?]', paragraph.strip()) if len(item.strip()) > 3]
        paragraph_count += 1
    data = {"question": question, 'infos': infos}
    # if sum([sentence_label[item][sentence] for item in sentence_label for sentence in sentence_label[item]]) != len(answers):
    #     new_answers = []
    #     for answer in answers:
    #         new_answers.append(answer)
    #         find = False
    #         for i, sents in sentences.items():
    #             if find:
    #                 break
    #             for sentence in sents:
    #                 if sentence == answer:
    #                     find = True
    #                     break
    #                 elif sentence.find(answer) != -1:
    #                     print(f"{answer_file} : {answer} : {sentence}")
    #                     find = True
    #                     new_answers.pop(-1)
    #                     new_answers.append(sentence)
            # with open(answer_file + '_', mode='w') as f:
            #     for sent in new_answers:
            #         f.write(sent + '\n')
        # return data
    return data
    
if __name__ == '__main__':
    data_list = []
    for tag in folders:
        sub_folders = folders[tag]
        for folder in sub_folders:
            if folder.endswith('.DS_Store'):
                continue
            data = construct_sample(folder + '/问题.txt', folder + '/材料.txt', folder + '/答案.txt')
            data['tag'] = tag
            data_list.append(data)
    with open(save_path, mode='w') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        
