
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# In[ ]:


class ESIM(nn.Module):
    def __init__(self, embedding_dim,dim_hidden,batch_size,vocab_size,embeddings):
        super(ESIM, self).__init__()
        self.batch_size=batch_size
        self.dim_hidden=dim_hidden
        if torch.cuda.is_available():
            self.embed = nn.Embedding(vocab_size,embedding_dim).cuda()
            self.embed.weight = nn.Parameter(torch.Tensor(embeddings).cuda())
            self.dropout = nn.Dropout(p=0.1).cuda()
            self.fc1 = nn.Linear(dim_hidden*8,dim_hidden).cuda()
            self.fc2 = nn.Linear(dim_hidden*8,dim_hidden).cuda()
            self.fc_op=nn.Linear(dim_hidden,3).cuda()
            self.LSTM_encoder = nn.LSTM(input_size=embedding_dim, hidden_size=dim_hidden,batch_first=False).cuda()
            self.LSTM_encoder_rev = nn.LSTM(input_size=embedding_dim, hidden_size=dim_hidden,batch_first=False).cuda()
            self.LSTM_decoder = nn.LSTM(input_size=dim_hidden, hidden_size=dim_hidden,batch_first=False).cuda()
            self.LSTM_decoder_rev = nn.LSTM(input_size=dim_hidden, hidden_size=dim_hidden,batch_first=False).cuda()
        else:
            self.embed = nn.Embedding(vocab_size,embedding_dim)
            self.embed.weight = nn.Parameter(torch.Tensor(embeddings))
            self.dropout = nn.Dropout(p=0.1)
            self.fc1 = nn.Linear(dim_hidden*8,dim_hidden)
            self.fc2 = nn.Linear(dim_hidden*8,dim_hidden)
            self.fc_op=nn.Linear(dim_hidden,3)
            self.LSTM_encoder = nn.LSTM(input_size=embedding_dim, hidden_size=dim_hidden,batch_first=False)
            self.LSTM_encoder_rev = nn.LSTM(input_size=embedding_dim, hidden_size=dim_hidden,batch_first=False)
            self.LSTM_decoder = nn.LSTM(input_size=dim_hidden, hidden_size=dim_hidden,batch_first=False)
            self.LSTM_decoder_rev = nn.LSTM(input_size=dim_hidden, hidden_size=dim_hidden,batch_first=False)

    def reverseTensor(self, tensor):
        idx = [i for i in range(tensor.size(0)-1, -1, -1)]
        if torch.cuda.is_available():
            idx = Variable(torch.LongTensor(idx).cuda())
        else:
            idx = Variable(torch.LongTensor(idx))
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor


    def softmax(self,input, axis=1):
        input_size = input.size()

        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()

        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)

        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)
    def forward(self, x1, x1_mask, x2, x2_mask,l):
        n_timesteps_premise = x1.size(0)
        n_timesteps_hypothesis = x2.size(0)
        n_samples = x1.size(1)

        xr1=self.reverseTensor(x1)
        #xr1_mask=self.reverseTensor(x1_mask)
        xr2=self.reverseTensor(x2)
        #xr2_mask=self.reverseTensor(x2_mask)
        emb1=self.dropout(self.embed(x1))
        #embr1=self.reverseTensor(emb1)
        emb2=self.dropout(self.embed(x2))
        #embr2=self.reverseTensor(emb2)
        ctx1=self.BiLSTM_encoder(emb1,x1_mask)
        ctx2=self.BiLSTM_encoder(emb2,x2_mask)

        ctx1=ctx1.permute(1,0,2)
        ctx2 = ctx2.permute(1, 0, 2)
        premise_list = list(torch.split(ctx1, split_size=1,dim=1))
        premise_list =[p.squeeze(1) for p in premise_list]
        hypothesis_list = list(torch.split(ctx2, split_size=1,dim=1))
        hypothesis_list = [h.squeeze(1) for h in hypothesis_list]
        scores_all = []
        premise_attn = []
        alphas = []
        for i in range(n_timesteps_premise):
            scores_i_list = []
            for j in range(n_timesteps_hypothesis):
                score_ij = torch.max(torch.mul(premise_list[i], hypothesis_list[j]), 1, keepdim=True)[0]
                scores_i_list.append(score_ij)

            scores_i = torch.stack(scores_i_list, 1)
            #print("sci",scores_i_list[0].size(),scores_i.size())
            alpha_i = self.masked_softmax(scores_i, x2_mask)
            a_tilde_i = torch.max(torch.mul(alpha_i, ctx2), 1)[0]
            premise_attn.append(a_tilde_i)

            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = torch.stack(scores_all, 2)
        #print("scall",scores_all[0].size(),scores_stack.size())
        scores_list = list(torch.split(scores_stack, split_size=1,dim=1))
        scores_list = [s.squeeze(1) for s in scores_list]
        #print("sl",scores_list[0].size())
        hypothesis_attn = []
        betas = []
        for j in range(len(scores_list)):
            scores_j = scores_list[j]
            beta_j = self.masked_softmax(scores_j, x1_mask)
            #print("beta_j",beta_j.size())
            b_tilde_j = torch.sum(torch.mul(beta_j, ctx1), 1)
            #print("b_tilde_j",b_tilde_j.size())
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        #print(premise_attn[0].size(),hypothesis_attn[0].size())
        # Make attention-weighted sentence representations into one tensor,
        premise_attn=[p.unsqueeze(1) for p in premise_attn]
        hypothesis_attn=[h.unsqueeze(1) for h in hypothesis_attn]

        ctx1_ = torch.cat(premise_attn, 1)
        ctx2_ = torch.cat(hypothesis_attn, 1)
        #print(ctx2_.size())
        ctx1 = ctx1.permute(1, 0, 2)
        ctx2 = ctx2.permute(1, 0, 2)
        ctx1_ = ctx1_.permute(1, 0, 2)
        ctx2_ = ctx2_.permute(1, 0, 2)

        inp1 = torch.cat([ctx1, ctx1_, ctx1*ctx1_, ctx1-ctx1_], 2)
        inp2 = torch.cat([ctx2, ctx2_, ctx2*ctx2_, ctx2-ctx2_], 2)
        #print("inp1 size",inp1.size())
        inp1 = self.dropout(F.relu(self.fc1(inp1)))
        inp2=self.dropout(F.relu(self.fc1(inp2)))

        ctx3=self.BiLSTM_decoder(inp1)
        ctx4=self.BiLSTM_decoder(inp2)

        logit1 = (ctx3 * x1_mask.unsqueeze(2)).sum(0) / x1_mask.sum(0).unsqueeze(1)
        logit2 = (ctx3 * x1_mask.unsqueeze(2)).max(0)[0]
        logit3 = (ctx4 * x2_mask.unsqueeze(2)).sum(0) / x2_mask.sum(0).unsqueeze(1)
        logit4 = (ctx4 * x2_mask.unsqueeze(2)).max(0)[0]
        logit = torch.cat([logit1, logit2, logit3, logit4], 1)
        logit=self.dropout(logit)
        logit=self.dropout(F.tanh(self.fc2(logit)))
        logit=self.fc_op(logit)
        #probs=F.softmax(logit)
        return logit

    def BiLSTM_encoder(self,x,mask):
        b=x.size()[1]
        xr=self.reverseTensor(x)
        if torch.cuda.is_available():
            h0 = c0 = Variable(torch.zeros((1,b,self.dim_hidden))).cuda()
            hr0 = cr0 = Variable(torch.zeros((1,b,self.dim_hidden))).cuda()
        else:
            h0 = c0 = Variable(torch.zeros((1, b, self.dim_hidden)))
            hr0 = cr0 = Variable(torch.zeros((1, b, self.dim_hidden)))
        self.LSTM_encoder.flatten_parameters()
        _,(proj,_)=self.LSTM_encoder(x,(h0,c0))
        self.LSTM_encoder_rev.flatten_parameters()
        _,(projr,_)=self.LSTM_encoder_rev(xr,(hr0,cr0))
        ctx=torch.cat((proj[0],self.reverseTensor(projr[0])),1)
        ctx=ctx*mask.unsqueeze(2)
        return ctx
    def BiLSTM_decoder(self,x):
        b=x.size()[1]
        xr=self.reverseTensor(x)
        if torch.cuda.is_available():
            h0 = c0 = Variable(torch.zeros((1,b,self.dim_hidden))).cuda()
            hr0 = cr0 = Variable(torch.zeros((1,b,self.dim_hidden))).cuda()
        else:
            h0 = c0 = Variable(torch.zeros((1, b, self.dim_hidden)))
            hr0 = cr0 = Variable(torch.zeros((1, b, self.dim_hidden)))
        self.LSTM_decoder.flatten_parameters()
        _,(proj,_)=self.LSTM_decoder(x,(h0,c0))
        self.LSTM_decoder_rev.flatten_parameters()
        _,(projr,_)=self.LSTM_decoder_rev(xr,(hr0,cr0))
        ctx=torch.cat((proj[0],self.reverseTensor(projr[0])),1)
        return ctx


    def masked_softmax(self,scores, mask):
        """
        Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.
        Input shape: (batch_size, max_seq_length, hidden_dim).
        mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
        """
        x=torch.exp((scores - torch.max(scores, 1, keepdim=True)[0])).squeeze()
        y=mask.permute(1,0)
        numerator = torch.mul(x,y).unsqueeze(2)
        #print("x,y",x.size(),y.size())
        #print("n",numerator.size())
        denominator = torch.sum(numerator, 1, keepdim=True)
        weights = torch.div(numerator, denominator)
        return weights

