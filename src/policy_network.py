import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    '''
    PolicyNetwork

    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()


        self.sent_group_fc = nn.Linear(input_dim, hidden_dim)

        self.sent_group_fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):


        sent_group_pred = self.sent_group_fc2(F.relu(self.sent_group_fc(x)))

        return sent_group_pred