#encoding: utf-8
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .baseRNN import BaseRNN
from .attention_1 import Attention_1

class StackRNN(object):
    def __init__(self, cell, initial_state, dropout, get_output, p_empty_embedding=None):
        self.cell = cell
        self.dropout = dropout
        self.s = [(initial_state, initial_state)]
        self.empty = initial_state
        self.get_output = get_output
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def push(self, expr, extra=None):
        #print(self.s[-1][0].size())
        #self.dropout(self.s[-1][0])
        self.s.append((self.cell(expr, self.s[-1][0]), extra))
        #print('stackLen' + str(len(self.s)))

    def pop(self):
        #print('stackLen' + str(len(self.s)-1))
        return self.s.pop()[1]


    def embedding(self):
        return self.get_output(self.s[-1][0]) if len(self.s) > 1 else self.empty

    def back_to_init(self):
        while self.__len__() > 0:
            self.pop()

    def clear(self):
        self.s.reverse()
        self.back_to_init()

    def __len__(self):
        return len(self.s) - 1

class DecoderRNN_3(BaseRNN):
    def __init__(self, vocab_size, class_size, embed_model=None, emb_size=100, hidden_size=128, \
                 n_layers=1, rnn_cell = None, rnn_cell_name='lstm', \
                 sos_id=1, eos_id=0, input_dropout_p=0, dropout_p=0): #use_attention=False):
        super(DecoderRNN_3, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p,
              n_layers, rnn_cell_name)
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding_size = emb_size
        self.dropout = dropout_p
        self.gpu_triger = True

        if embed_model == None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model

        if rnn_cell == None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers, \
                                 batch_first=True, dropout=dropout_p)
        else:
            self.rnn = rnn_cell

        self.out = nn.Linear(self.hidden_size * 3, self.class_size)
        self.add = nn.Linear(self.embedding_size * 2, self.embedding_size, bias = False)
        self.minus = nn.Linear(self.embedding_size * 2, self.embedding_size, bias = False)
        self.multiply = nn.Linear(self.embedding_size * 2, self.embedding_size, bias = False)
        self.divide = nn.Linear(self.embedding_size * 2, self.embedding_size, bias = False)
        self.power = nn.Linear(self.embedding_size * 2, self.embedding_size, bias = False)
        self.buffer_liner = nn.Linear(self.embedding_size * 2, self.hidden_size)
        self.attention = Attention_1(hidden_size)
        self.empty_emb = nn.Parameter(torch.randn(1, hidden_size))

        self.lstm_initial = (
            self.xavier_init(self.gpu_triger, 1, hidden_size),
            self.xavier_init(self.gpu_triger, 1, hidden_size))

        self.stack_lstm = nn.LSTMCell(self.embedding_size, hidden_size)

    #def _init_state(self, encoder_hidden, op_type):
        

    def forward_step(self, stack, input_var, hidden, encoder_outputs, buffer_emb, function):
        '''
        normal forward, step by step or all steps together
        '''
        if len(input_var.size()) == 1:
            input_var = torch.unsqueeze(input_var,1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        output, attn = self.attention(output, encoder_outputs)
        
        output = torch.cat((output, buffer_emb), 0)
        output = torch.cat((output, stack.embedding().unsqueeze(0)),0)

        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size*3)), dim=1)\
                            .view(batch_size, output_size, -1)
        return predicted_softmax, hidden#, #attn

    def decode(self, step, step_output):
        '''
        step_output: batch x classes , prob_log
        symbols: batch x 1
        '''
        symbols = step_output.topk(1)[1] 
        return symbols

    def decode_rule(self, step, sequence_symbols_list, step_output):
        symbols = self.rule_filter(sequence_symbols_list, step_output)
        return symbols

    def _rnn_get_output(self, state):
        return state[0]

    def variable(self, tensor, gpu):
        if gpu:
            return torch.autograd.Variable(tensor).cuda()
        else:
            return torch.autograd.Variable(tensor)

    def xavier_init(self, gpu, *size):
        return nn.init.xavier_normal(self.variable(torch.FloatTensor(*size), gpu))

    def buffer_rep(self, buffer):
        all_emb = None
        for index, item in enumerate(buffer.keys()):
            word = Variable(torch.LongTensor([self.vocab_dict[item]]))
            word_emb = self.embedding(word.cuda())
            count = 'count_'+str(min(9,buffer[item]))
            count = Variable(torch.LongTensor([self.vocab_dict[count]]))
            count_emb = self.embedding(count.cuda())
            embedding = torch.cat((word_emb, count_emb), 1)
            embedding = self.buffer_liner(embedding)

            if index == 0:
                all_emb = embedding
            else:
                all_emb = torch.cat((all_emb, embedding), 0)
        all_emb = all_emb.unsqueeze(0)
        output = torch.nn.functional.adaptive_avg_pool2d(all_emb, (1,self.hidden_size))
        return output

        
    def forward_normal_teacher_1(self, decoder_inputs, decoder_init_hidden, function):
        '''
        decoder_input: batch x seq_lengths x indices( sub last(-1), add first(sos_id))
        decoder_init_hidden: processed considering encoder layers, bi 
            lstm : h_0 (num_layers * num_directions, batch, hidden_size)
                   c_0 (num_layers * num_directions, batch, hidden_size)
            gru  : 
        decoder_outputs: batch x seq_lengths x classes,  probility_log
            lstm : h_n (num_layers * num_directions, batch, hidden_size)
                   c_n (num_layers * num_directions, batch, hidden_size)
            gru  :
        decoder_hidden: layers x batch x hidden_size 
        '''
        decoder_outputs, decoder_hidden = self.forward_step(\
                          decoder_inputs, decoder_init_hidden, function=function)
        decoder_outputs_list = []
        sequence_symbols_list = []
        for di in range(decoder_outputs.size(1)):
            step_output = decoder_outputs[di, :]
            symbols = self.decode(di, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_normal_teacher(self, decoder_inputs, decoder_init_hidden, encoder_outputs, buffer, function):
        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden
        seq_len = decoder_inputs.size(1)
        stack = StackRNN(self.stack_lstm, self.lstm_initial, self.dropout, self._rnn_get_output, self.empty_emb)
        for di in range(seq_len):
            decoder_input = decoder_inputs[:, di]
            #deocder_input = torch.unsqueeze(decoder_input, 1)
            #print '1', deocder_input.size()
            buffer_emb = self.buffer_rep(buffer)
            decoder_output, decoder_hidden = self.forward_step(\
                stack, decoder_input, decoder_hidden, encoder_outputs, buffer_emb, function=function)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            if self.use_rule == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
            symbols_str = self.class_list[symbols]
            print(symbols_str)
            if symbols_str.startswith('reduce'):
                op1 = stack.pop()
                op2 = stack.pop()
                if symbols_str.split('_')[1] == '+':
                    stack.push(self.add(torch.cat((op1,op2),1)), self.add(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '-':
                    stack.push(self.minus(torch.cat((op1,op2),1)), self.minus(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '*':
                    stack.push(self.multiply(torch.cat((op1,op2),1)), self.multiply(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '/':
                    stack.push(self.divide(torch.cat((op1,op2),1)), self.divide(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '^':
                    stack.push(self.power(torch.cat((op1,op2),1)), self.power(torch.cat((op1,op2),1)))
            elif symbols_str != 'END_token':
                stack.push(self.embedding(symbols).squeeze(0), self.embedding(symbols).squeeze(0))
                if symbols_str in buffer.keys():
                    buffer[symbols_str] = buffer[symbols_str] + 1
                else:
                    buffer['UNK_token'] = buffer['UNK_token'] + 1

        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list

    def symbol_norm(self, symbols):
        symbols = symbols.view(-1).data.cpu().numpy() 
        new_symbols = []
        for idx in symbols:
            #print idx, 
            #print self.class_list[idx],
            #pdb.set_trace()
            #print self.vocab_dict[self.class_list[idx]]
            try:
            	new_symbols.append(self.vocab_dict[self.class_list[idx]])
            except:
            	print("idx",idx)
            	
        new_symbols = self.embedding(Variable(torch.LongTensor(new_symbols)))
        #print new_symbols
        new_symbols = torch.unsqueeze(new_symbols, 0)
        if self.use_cuda:
            new_symbols = new_symbols.cuda()
        return new_symbols


    def forward_normal_no_teacher(self, decoder_input, decoder_init_hidden, encoder_outputs, buffer,\
                                                 max_length,  function):
        '''
        decoder_input: batch x 1
        decoder_output: batch x 1 x classes,  probility_log
        '''
        decoder_outputs_list = []
        sequence_symbols_list = []
        #attn_list = []
        decoder_hidden = decoder_init_hidden
        stack = StackRNN(self.stack_lstm, self.lstm_initial, self.dropout, self._rnn_get_output, self.empty_emb)
        for di in range(max_length):
            buffer_emb = self.buffer_rep(buffer)
            decoder_output, decoder_hidden = self.forward_step(\
                           stack, decoder_input, decoder_hidden, encoder_outputs, buffer_emb, function=function)
            #attn_list.append(attn)
            step_output = decoder_output.squeeze(1)
            #print step_output.size()
            if self.use_rule == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output) 
            decoder_input = symbols
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
            symbols_str = self.class_list[symbols]
            print(symbols_str)
            if symbols_str.split('_')[0] == 'reduce':
                op1 = stack.pop()
                op2 = stack.pop()
                if symbols_str.split('_')[1] == '+':
                    stack.push(self.add(torch.cat((op1,op2),1)), self.add(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '-':
                    stack.push(self.minus(torch.cat((op1,op2),1)), self.minus(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '*':
                    stack.push(self.multiply(torch.cat((op1,op2),1)), self.multiply(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '/':
                    stack.push(self.divide(torch.cat((op1,op2),1)), self.divide(torch.cat((op1,op2),1)))
                elif symbols_str.split('_')[1] == '^':
                    stack.push(self.power(torch.cat((op1,op2),1)), self.power(torch.cat((op1,op2),1)))
            elif symbols_str != 'END_token':
                stack.push(self.embedding(symbols).squeeze(0), self.embedding(symbols).squeeze(0))
        #print sequence_symbols_list
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list#, attn_list


    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, buffer = None, template_flag=True,\
                function=F.log_softmax, teacher_forcing_ratio=0, use_rule=True, use_cuda=False, \
                vocab_dict = None, vocab_list = None, class_dict = None, class_list = None):
        '''
        使用rule的时候，teacher_forcing_rattio = 0
        '''
        self.use_rule = use_rule
        self.use_cuda = use_cuda
        self.class_dict = class_dict
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        all_encoder_outputs = []
        all_decoder_hidden = []
        all_decoder_cell = []
        all_sequence_symbols_list = []
        print('inputs: '+str(len(inputs)))
        print(' hidden:' + str(len(encoder_hidden)))
        print(' outputs:' + str(len(encoder_outputs)))
        print('buffer:'+ str(len(buffer)))
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #pdb.set_trace()
        batch_size = encoder_outputs.size(0)
        #batch_size = inputs.size(0)
        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker

        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.use_cuda:
            pad_var = pad_var.cuda()

        decoder_init_hidden = encoder_hidden

        if template_flag == False:
            max_length = 40
        else:
            max_length = inputs.size(1)
        #inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
        #inputs = inputs[:, :-1] # batch x seq_len
        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
            inputs = inputs[:, :-1] # batch x seq_len
            decoder_inputs = inputs
            for i in range(batch_size):
                encoder_outputs_list, decoder_hidden, sequence_symbols_list = self.forward_normal_teacher(\
                    decoder_inputs[i].unsqueeze(0),(decoder_init_hidden[0][:,i,:].unsqueeze(1).contiguous(),decoder_init_hidden[1][:,i,:].unsqueeze(1).contiguous()), encoder_outputs[i].unsqueeze(0), buffer[i], function)
                all_encoder_outputs.append(encoder_outputs_list)
                all_decoder_hidden.append(decoder_hidden[0])
                all_decoder_cell.append(decoder_hidden[1])
                all_sequence_symbols_list.append(sequence_symbols_list)
            all_decoder_hidden = torch.cat(all_decoder_hidden, 0)
            all_decoder_cell = torch.cat(all_decoder_cell,0)
            outputs_list = []
            symbols_list = []
            for di in range(max_length):
                tmp_outputs = all_encoder_outputs[0][di]
                tmp_symbols = all_sequence_symbols_list[0][di]
                for bi in range(batch_size - 1):
                    tmp_outputs = torch.cat((tmp_outputs, all_encoder_outputs[bi + 1][di]), 0)
                    tmp_symbols = torch.cat((tmp_symbols, all_sequence_symbols_list[bi + 1][di]), 0)
                outputs_list.append(tmp_outputs)
                symbols_list.append(tmp_symbols)
            return outputs_list, (all_decoder_hidden,all_decoder_cell), symbols_list

        else:
            #decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_inputs = pad_var#.unsqueeze(1) # batch x 1
            #pdb.set_trace()
            for i in range(batch_size):
                try:
                    encoder_outputs_list, decoder_hidden, sequence_symbols_list = self.forward_normal_no_teacher(\
                        decoder_inputs[i].unsqueeze(0),(decoder_init_hidden[0][:,i,:].unsqueeze(1).contiguous(),decoder_init_hidden[1][:,i,:].unsqueeze(1).contiguous()), encoder_outputs[i].unsqueeze(0), buffer[i], max_length, function)
                except IndexError:
                    print('i:' + str(i) + 'inputs:' + str(len(decoder_inputs)))
                    print('buffer:' + str(len(buffer)))
                else:
                    all_encoder_outputs.append(encoder_outputs_list)
                    all_decoder_hidden.append(decoder_hidden[0])
                    all_decoder_cell.append(decoder_hidden[1])
                    all_sequence_symbols_list.append(sequence_symbols_list)
            all_decoder_hidden = torch.cat(all_decoder_hidden, 0)
            all_decoder_cell = torch.cat(all_decoder_cell, 0)
            outputs_list = []
            symbols_list = []
            for di in range(max_length):
                tmp_outputs = all_encoder_outputs[0][di]
                tmp_symbols = all_sequence_symbols_list[0][di]
                for bi in range(len(all_encoder_outputs)-1):
                    tmp_outputs = torch.cat((tmp_outputs, all_encoder_outputs[bi+1][di]), 0)
                    tmp_symbols = torch.cat((tmp_symbols, all_sequence_symbols_list[bi+1][di]), 0)
                outputs_list.append(tmp_outputs)
                symbols_list.append(tmp_symbols)
            return outputs_list,(all_decoder_hidden,all_decoder_cell), symbols_list


    def rule(self, symbol):
        filters = []
        if self.class_list[symbol].split('_')[1] in ['+', '-', '*', '/']:
            filters.append(self.class_dict['reduce_+'])
            filters.append(self.class_dict['reduce_-'])
            filters.append(self.class_dict['reduce_*'])
            filters.append(self.class_dict['reduce_/'])
            filters.append(self.class_dict['reduce_)'])
            filters.append(self.class_dict['='])
        elif self.class_list[symbol] == '=':
            filters.append(self.class_dict['reduce_+'])
            filters.append(self.class_dict['reduce_-'])
            filters.append(self.class_dict['reduce_*'])
            filters.append(self.class_dict['reduce_/'])
            filters.append(self.class_dict['='])
            filters.append(self.class_dict['reduce_)'])
        elif self.class_list[symbol] == 'reduce_(':
            filters.append(self.class_dict['reduce_('])
            filters.append(self.class_dict['reduce_)'])
            filters.append(self.class_dict['reduce_+'])
            filters.append(self.class_dict['reduce_-'])
            filters.append(self.class_dict['reduce_*'])
            filters.append(self.class_dict['reduce_/'])
            filters.append(self.class_dict['=']) 
        elif self.class_list[symbol] == 'reduce_)':
            filters.append(self.class_dict['reduce_('])
            filters.append(self.class_dict['reduce_)'])
            for k,v in self.class_dict.items():
                if 'temp' in k:
                    filters.append(v)
        elif 'temp' in self.class_list[symbol]:
            filters.append(self.class_dict['reduce_('])
            filters.append(self.class_dict['=']) 
        return np.array(filters)

    def filter_op(self):
        filters = []
        filters.append(self.class_dict['reduce_+'])
        filters.append(self.class_dict['reduce_-'])
        filters.append(self.class_dict['reduce_*'])
        filters.append(self.class_dict['reduce_/'])
        filters.append(self.class_dict['reduce_^'])
        return np.array(filters)

    def filter_END(self):
        filters = []
        filters.append(self.class_dict['END_token']) 
        return np.array(filters)

    def filter_outlier(self):
        filters = []
        filters.append(self.class_dict['END_token'])
        return np.array(filters)
        

    def rule_filter(self, sequence_symbols_list, current):
        '''
        32*28
        '''
        op_list = ['+','-','*','/','^']
        cur_out = current.cpu().data.numpy()
        #print len(sequence_symbols_list)
        #pdb.set_trace()
        cur_symbols = []
        if sequence_symbols_list == [] or len(sequence_symbols_list) <= 1:
            #filters = self.filter_op()
            filters = np.append(self.filter_op(), self.filter_END())
            cur_out[0][filters] = -float('inf')
            cur_symbols.append(np.argmax(cur_out))
        else:
            num_var = 0
            num_op = 0
            for j in range(len(sequence_symbols_list)):
                symbol = sequence_symbols_list[j].cpu().data[0]
                if '_' in self.class_list[symbol] and self.class_list[symbol].split('_')[1] in op_list:
                    num_op += 1
                elif 'shift' in self.class_list[symbol] or self.class_list[symbol] in ['1', 'PI']:
                    num_var += 1
            print('num: '+ str(num_var) +' op:'+str(num_op))
            if num_var >= num_op + 2:
                filters = self.filter_END()
                cur_out[0][filters] = -float('inf')
            if num_var <= num_op + 1:
                filters = self.filter_op()
                cur_out[0][filters] = -float('inf')
            cur_symbols.append(np.argmax(cur_out))
        cur_symbols = Variable(torch.LongTensor(cur_symbols))
        cur_symbols = torch.unsqueeze(cur_symbols, 0)
        cur_symbols = cur_symbols.cuda()
        return cur_symbols

