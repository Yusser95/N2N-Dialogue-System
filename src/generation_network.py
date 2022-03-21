import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from src.utils import tokenize

class ConditionNet(nn.Module):
    '''
    Generate condition vector for conditional generation

    cond = [sent_vec, state_rep, slot_states, kb_found]
    '''
    def __init__(self, sent_vec_size, state_tracker_hidden_size, slot_states_len, kb_found_size, cond_size):
        super(ConditionNet, self).__init__()
        self.sent_vec_size = sent_vec_size
        self.state_tracker_hidden_size = state_tracker_hidden_size
        self.slot_states_len = slot_states_len
        self.kb_found_size = kb_found_size
        self.cond_size = cond_size

        self.input_size = self.sent_vec_size + self.state_tracker_hidden_size + self.slot_states_len + self.kb_found_size

        self.fc = nn.Linear(self.input_size, self.cond_size)
        self.bn = nn.BatchNorm1d(self.cond_size)
    
    def forward(self, sent_vec, state_rep, slot_states, kb_found):
        slot_states_rep = torch.cat([ slot_states[slot] for slot in sorted(slot_states.keys())], dim=1)
        slot_states_rep = slot_states_rep.view(state_rep.size(0), -1).float()

        inp = torch.cat([sent_vec, state_rep, slot_states_rep, kb_found.float()], dim=1)
       
        assert(inp.size(1) == self.input_size)

        return F.tanh(self.bn(self.fc(inp)))


class Generator(nn.Module):
    '''
    Generate output sequence distribution given last time input and condition
    '''
    def __init__(self, cond_size, hidden_size, vocab_size, embed_dim):
        super(Generator, self).__init__()

        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=self.cond_size + self.vocab_dim, \
                            hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inp, cond, hidden):
        # inp = [[idx, ]]
        # cond = C

        embed = self.embedding(inp) # (1, N words, vocab_dim)

        x = torch.cat([embed, cond.view(1,1,-1)], dim=2).view(1, -1, self.cond_size + self.vocab_dim)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

def load_generator_model(conf, cond_net_fn, gen_fn, slot_len_sum, sys_vocab_size):
    cond_net = ConditionNet(conf["sentvec_size"], conf["state_tracker_hidden_size"],
                    slot_len_sum, conf["kb_indicator_len"], conf["cond_size"])
    generator = Generator(conf["cond_size"], conf["generator_hidden_size"], sys_vocab_size, conf["sys_embed_dim"])

    with open(cond_net_fn, 'rb') as f:
        cond_net.load_state_dict(torch.load(f))

    with open(gen_fn, 'rb') as f:
        generator.load_state_dict(torch.load(f))

    return cond_net, generator

class NaiveDecoder:
    def __init__(self, vocabs):
        self.vocabs = vocabs

    def decode(self, sent_out):
        '''
        out : N words * vocab_size (p distribution)
        '''
        sent = []

        for w in sent_out:
            sent.append(self.vocabs[w])

        return " ".join(sent)


class Codec:
    def __init__(self, words, word2idx, decoder=NaiveDecoder, tokenize=tokenize):
        self.words = words
        self.word2idx = word2idx
        self.decoder = decoder(words)
        self.tokenize = tokenize

    def encode(self, sent):
        tokens = self.tokenize(sent.lower())
        return [ self.word2idx.get(t, 0) for t in tokens ]

    def decode(self, seq):
        return self.decoder.decode(seq)



class SentenceGenerator:
    def __init__(self, kb, onto, sent_groups):
        self.last_states_pred_dict = None
        self.last_selected_search_result = None
        self.kb = kb
        self.onto = onto
        self.sent_groups = sent_groups
        self.sent_type_mapping = {
            "132":"137",
            "136":"137",
            "138":"137",
            "24":"41",
            "20":"137",
            "161":"41",
            # "27":"137",
            "21":"41",
            "8":"137",
            "96":"41",
            "120":"41",
            "112":"150",
            "122":"137",
            "123":"137",
            "124":"137",
            "126":"41",
            "194":"137",
            "197":"137",
            "196":"137",
            "191":"137",
            "193":"137",
            "115":"137",
            "117":"137",
            "116":"137",
            "111":"41",
            "110":"41", # An interesting group ...
            "176":"137", # Another interesting group
            "82":"137",
            "86":"137",
            "118":"137",
            "178":"137",
            "108":"137",
            "109":"137",
            "103":"137",
            "100":"137",
            "30":"137",
            "37":"41",
            "35":"137",
            # "34":"137",
            "60":"137",
            "65":"137",
            "68":"137",
            "175":"137",
            "173":"137",
            "171":"137",
            "170":"137",
            "182":"137",
            "183":"137",
            "180":"137",
            "181":"137",
            "6":"137",
            "99":"137",
            "163":"137",
            "15":"137",
            # "14":"137",
            "17":"137",
            "152":"137",
            "158":"41",
            "78":"41",
            "148":"137",
            "144":"137",

        }

    def generate(self, states_pred_dict, sent_type):
        # the index of onto is 1 greater than argmax
        sentence = ""
        # possible search fields: area, food, and pricerange
        search_query = []
        search_result = []
        selected_search_result = ""
        print (sent_type)
        if sent_type in self.sent_type_mapping.keys():
            sent_type = self.sent_type_mapping[sent_type]
        print (sent_type)
        original_sent = random.choice(self.sent_groups[sent_type])
        original_words = original_sent.split(" ")

        for key, value in states_pred_dict.items():
            if key == "food":
                record_food_type = self.onto.get(key)[value.item() - 1]
            if key == "area":
                #print(value,key)
                record_area_type = self.onto.get(key)[value.item() - 1]
            if key == "pricerange":
                record_pricerange_type = self.onto.get(key)[value.item() - 1]

        if self.last_states_pred_dict is not None \
                and self.last_states_pred_dict is not None \
                and states_pred_dict.get("area").item() == self.last_states_pred_dict.get("area").item() \
                and states_pred_dict.get("food").item() == self.last_states_pred_dict.get("food").item() \
                and states_pred_dict.get("pricerange").item() == self.last_states_pred_dict.get("pricerange").item():
            selected_search_result = self.last_selected_search_result
        else:
            for key, value in states_pred_dict.items():
                if not key.endswith("_req") and value.item() != 0:
                    search_query.append([key, self.onto.get(key)[value.item() - 1]])
            search_result = list(self.kb.search_multi(search_query))
            print (search_query)
            print (search_result)
            if len(search_result) != 0:
                search_result_length = len(search_result)
                selected_search_result = search_result[random.randint(0,search_result_length - 1)]
                self.last_states_pred_dict = states_pred_dict
                self.last_selected_search_result = selected_search_result
                print (self.kb.get(selected_search_result))
            elif len(search_result) == 0:
                self.last_selected_search_result = None
                self.last_selected_search_result = None
                original_sent = random.choice(self.sent_groups[str(int(41))])
                original_words = original_sent.split(" ")
        for original_word in original_words:
            if original_word.startswith("<v.ADDRESS>"):
                sentence = sentence + self.kb.get(selected_search_result).get('address') + " "
            elif original_word.startswith("<v.AREA>"):
                if len(search_result) == 0:
                    sentence = sentence + record_area_type + " "
                else:
                    sentence = sentence + self.kb.get(selected_search_result).get('area') + " "
            elif original_word.startswith("<v.FOOD>"):
                if len(search_result) == 0:
                    sentence = sentence + record_food_type + " "
                else:
                    sentence = sentence + self.kb.get(selected_search_result).get('food') + " "
            elif original_word.startswith("<v.NAME>"):
                sentence = sentence + self.kb.get(selected_search_result).get('name') + " "
            elif original_word.startswith("<v.PHONE>"):
                sentence = sentence + self.kb.get(selected_search_result).get('phone') + " "
            elif original_word.startswith("<v.POSTCODE>"):
                sentence = sentence + self.kb.get(selected_search_result).get('postcode') + " "
            elif original_word.startswith("<v.PRICERANGE>"):
                if len(search_result) == 0:
                    sentence = sentence + record_pricerange_type + " "
                else:
                    sentence = sentence + self.kb.get(selected_search_result).get('pricerange') + " "
            elif original_word.startswith("<s.ADDRESS>"):
                sentence = sentence + "address "
            elif original_word.startswith("<s.AREA>"):
                sentence = sentence + "area "
            elif original_word.startswith("<s.FOOD>"):
                sentence = sentence + "food "
            elif original_word.startswith("<s.NAME>"):
                sentence = sentence + "name "
            elif original_word.startswith("<s.PHONE>"):
                sentence = sentence + "phone "
            elif original_word.startswith("<s.POSTCODE>"):
                sentence = sentence + "postcode "
            elif original_word.startswith("<s.PRICERANGE>"):
                sentence = sentence + "pricerange "
            elif original_word == "ly":
                sentence = sentence.strip() + "ly "
            else:
                sentence = sentence + original_word + " "
        return sentence


if __name__ == '__main__':
    g = Generator(10, 11, 12, 14)

    print (g(Variable(torch.LongTensor([[2]])), Variable(torch.zeros(1, 1, 10)), g.init_hidden()))

