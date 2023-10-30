import pickle
from statistics import mean


class AutoDict(dict):
    def __missing__(self, k):
        self[k] = AutoDict()
        return self[k]

Dataset = 'Thumos'



with open(Dataset+'/'+Dataset+'_features_seq_len_12_.pkl','rb') as ff:
    feat_json = pickle.load(ff)

n_of_agents = []

for vid in feat_json['database']:
    for snips in feat_json['database'][vid]['snippets']:
        # print('snip',len(feat_json['database'][vid]['snippets']))
        n_of_agents.append(len(feat_json['database'][vid]['snippets'][snips]['agents'])+1)
        # print(len(feat_json['database'][vid]['snippets'][snips]['agents']))
        # for agents in feat_json['database'][vid]['snippets'][snips]['agents']:

            # n_of_agents.append(agents)
print(mean(n_of_agents))


    

