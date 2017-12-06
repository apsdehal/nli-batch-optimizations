
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
    def bmm(self,a,b):
        new_tensor=[]
        for i in range(0,a.size(0)):
            new_tensor.append(torch.matmul(a[i],b[i]).unsqueeze(0))
        new_tensor=torch.cat(new_tensor,0)
        return new_tensor
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

        #Alignment layer
        weight_matrix = self.bmm(ctx1.permute(1,0,2), ctx2.permute(1,2,0))
        weight_matrix_1 = torch.exp(weight_matrix - torch.max(weight_matrix,1, keepdim=True)[0] ).permute(1,2,0)
        weight_matrix_2 = torch.exp(weight_matrix - torch.max(weight_matrix,2, keepdim=True)[0]).permute(1,2,0)

        weight_matrix_1 = weight_matrix_1 * x1_mask.unsqueeze(1)
        weight_matrix_2 = weight_matrix_2 * x2_mask.unsqueeze(0)
        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)

        ctx2_ = (ctx1.unsqueeze(1) * alpha.unsqueeze(3)).sum(0)
        ctx1_ = (ctx2.unsqueeze(0) * beta.unsqueeze(3)).sum(1)

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
        probs=F.softmax(logit)
        return probs

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

