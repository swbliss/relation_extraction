# Let's analyze why relation has 4 element. (assume: an entity pair can have multiple relation limited by 4)
# Lets see the entities and sample sentences showing same relation (e.x. 5 2 -1 -1)
# I think -1 means nothing, 5, 2, means some relation.

target_relation = ["4", "3", "-1", "-1"]

word_file = open("./gap_40_len_80/50/dict.txt")
word_dic = [""]
idx = 0
while True:
    word = word_file.readline()
    if not word: break
    word_dic.append(word.strip())
word_dic.append("NONE")

train_data = open("./gap_40_len_80/train_filtered.data")
train_data_for_relation = open(
    "./gap_40_len_80/train_for_relation_" + "_".join(target_relation) + ".txt", "w")
instances_for_relation = 1      # training instance for relation 5 2 -1 -1

def index_to_word(idx_str):
    return " ".join(map(lambda x: word_dic[int(x)], idx_str.split(" ")))

relation_types = {}
while True:
    entities = train_data.readline()
    if not entities: break
    relation_line_list = train_data.readline().split(" ")
    relation = relation_line_list[:-1]
    rel_name = relation[0]
    sent_num = int(relation_line_list[-1])
    if not rel_name in relation_types:
        relation_types[rel_name] = 1
    else:
        relation_types[rel_name] += sent_num
    for i in range(sent_num):
        sent = train_data.readline()
        if relation == target_relation:
            if i == 0:
                train_data_for_relation.write(index_to_word(entities) + "\n")
            train_data_for_relation.write(index_to_word(sent) + "\n")

    if relation == target_relation:
        train_data_for_relation.write("\n")

import operator
sorted_relation_types = sorted(relation_types.items(), key=operator.itemgetter(1))

count_list = []
for item in sorted_relation_types:
    count_list.append(item[1])
print(count_list)
word_file.close()
train_data.close()
train_data_for_relation.close()




