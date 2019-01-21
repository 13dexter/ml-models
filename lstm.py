import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math

_VF = torch._C._VariableFunctions

# Unoptimized implementation
# Directly running on cuda will not speed it up. Needs batching.

class LstmModule(nn.Module):
    def __init__(self, input_units, output_units, hidden_units, bias = True, num_chunks = 4, embedding_dim = 50, rectify_inputs = True, var_input = 0.01**2, var_rec = 0.15**2, dt = 0.5, tau=100):
        super(LstmModule, self).__init__()

        input_size = input_units
        hidden_size = hidden_units
        num_chunks = 2
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.igate = nn.Parameter(torch.tensor(0.5))
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        print('1 :', self.igate)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        print('2 :', self.igate)
        nn.init.uniform_(self.igate, 0, 1)
        print('3 :', self.igate)

    def forward(self, input_, hx = None):
        """
            begin{array}{ll}
            i = sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
            f = sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
            g = tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
            o = sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
            c' = f * c + i * g \\
            h' = o * tanh(c') \\
            end{array}
        """
        if hx is None:
            hx = input_.new_zeros(self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        use_gate = rectify(self.igate)
        if (self.igate > 1.0) :
        #     use_gate = self.igate - 1.0
            print('ho')

        hprev, cprev = hx
        w_x = torch.addmv(self.bias_ih, self.weight_ih, input_)
        w_h = torch.addmv(self.bias_hh, self.weight_hh, hprev)
        w_w = w_x + w_h

        i = self.sigmoid(w_w[0 : self.hidden_size])
        f = self.sigmoid(w_w[self.hidden_size : 2*self.hidden_size])
        o = self.sigmoid(w_w[2*self.hidden_size : 3*self.hidden_size])
        g = self.tanh(w_w[3*self.hidden_size : 4*self.hidden_size])

        c = (f * cprev) + (i * g)
        h = o * self.tanh(c)

        return (h, c), o

class LSTM(nn.Module):
    def __init__(self, input_units, hidden_units, vocab_size, embedding_dim = 50, output_units = 10, num_layers = 1, dropout=0):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.num_layers = num_layers
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_units = input_units if layer == 0 else hidden_units
            cell = LstmModule(input_units = input_units, output_units = output_units, hidden_units = hidden_units)
            setattr(self, 'cell_{}'.format(layer), cell)
        
        self.embedding_layer = torch.nn.Embedding(vocab_size, input_units)
        # self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_units * num_layers, 2)
        # self.softmax = nn.Softmax(dim=0)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def forward(self, input_, max_time = 10, input_once = True, states_init = None) :
        if states_init is None:
            states_init = torch.zeros([self.hidden_units, self.embedding_dim], dtype=torch.float)
        # state_n = []
        layer_output = None
        all_layers_last_hidden = []
        input0 = torch.zeros(len(input_), dtype=torch.long)
        inputx = torch.tensor(input_, requires_grad = False)
        state = None
        max_time = len(input_)

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            for time in range(max_time):
                inputx = torch.tensor(input_[time], requires_grad = False)
                state, outs = cell(input_ = self.embedding_layer(inputx), hx = state)
        hlast, clast = state
        softmax_out = self.linear(hlast)
        softmax_out = torch.stack([softmax_out], 0)
        return softmax_out