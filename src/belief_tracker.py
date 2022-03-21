import torch
import torch.nn as nn
import torch.functional as F

from torch.autograd import Variable

class StateTracker(nn.Module):

    def __init__(self, feat_size, hidden_size):
        super(StateTracker, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=feat_size,
                            hidden_size=hidden_size, batch_first=True, dropout=0.9)

    def forward(self, inp, hidden):
        out, h_n = self.gru(inp, hidden)

        return out, h_n

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))



class InformSlotTracker(nn.Module):
    '''
        Informable Slot Tracker

        get slot value distribution from state at time t

        e.g. price=cheap

        input: state tracker output `state_t`
        output: value distribution `P(v_s_t| state_t)`
    '''
    def __init__(self, input_dim, n_choices):
        super(InformSlotTracker, self).__init__()
        self.n_choices = n_choices + 1 # include don't care
        self.fc = nn.Linear(input_dim, self.n_choices)

    def forward(self, state):
        return self.fc(state)

class RequestSlotTracker(nn.Module):
    '''
        Requestable Slot Tracker

        get a request type activation state distribution from state at time t

        e.g.
            address=1 (currently address is requested)
            phone=0 (currently phone is not cared by the user)

        input: state tracker output `state_t`
        output: value binary distribution `P(v_s_t| state_t)`
    '''
    def __init__(self, input_dim):
        super(RequestSlotTracker, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, state):
        return self.fc(state)
