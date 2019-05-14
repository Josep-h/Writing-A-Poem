import json
import jieba
from langconv import *
import random
import torch
from torch.autograd import Variable, Function
import numpy as np


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# parameters

dtype=torch.FloatTensor

lines_of_poem=4
char_each_line=12
D_input=300
D_output=500
D_batch=1
D_dict=0 # this is the dictionary of all word I have
# D_dict=1000 # the dimension of dictionary
turns=300
alpha=0.02

eight_lines=[]


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# build vector

# map the char into a float list which is its vec
dict={}

# store the words with ptrtrain
wordspool=[]

# for all the chars in my dict
dictionary={}

def build_dict():
    with open("embedding.word","r",encoding='UTF-8') as f:
        f.readline()
        while 1:
            line=f.readline()
            if not line:
                break
            split=line.split()
            char=split[0]

            wordspool.append(char)
            
            vec=[0]*D_input  #fot the dimension of input
            for i in range(0,D_input):
                vec[i]=float(split[i+1])
            dict[char]=vec
        f.close

# After this function, I wll get a poem which was transformed in a character list

def clr_poem(addr="test.json"):
    eight_lines=[]
    ct=0
    with open(addr,"r",encoding='UTF-8') as f:
        all = json.loads(f.read())
        for poem in all:
            content=poem['paragraphs']
            if len(content)==4:
                if len(content[0])==12:
                    # print(poem['title'])
                    code=[]
                    chars=[]
                    for line in content:    
                        line = Converter('zh-hans').convert(line)
                        # print(line)
                        for j in range(0,12):
                            in_char=line[j]
                            if line[j]=='，' or line[j]=='。':
                                in_char=','
                            if in_char not in dictionary.keys():
                                dictionary[in_char]=ct
                                ct+=1
                            chars.append(in_char)
                    eight_lines.append(chars)
    print(len(eight_lines))
    return eight_lines,ct



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# LSTM

class MyLSTM:
    def __init__(self,D_input,D_output,D_batch):
        self.D_input=D_input
        self.D_output=D_output
        self.D_batch=D_batch
        # bi,bf,bo,bc
        self.bi=Variable(torch.rand(self.D_output),requires_grad=True)
        self.bf=Variable(torch.rand(self.D_output),requires_grad=True)
        self.bo=Variable(torch.rand(self.D_output),requires_grad=True)
        self.bc=Variable(torch.rand(self.D_output),requires_grad=True)


        # Wi,Wf,Wo,Wc
        self.Wi=Variable(torch.rand(self.D_output,self.D_input),requires_grad=True)
        self.Wf=Variable(torch.rand(self.D_output,self.D_input),requires_grad=True)
        self.Wo=Variable(torch.rand(self.D_output,self.D_input),requires_grad=True)
        self.Wc=Variable(torch.rand(self.D_output,self.D_input),requires_grad=True)


        # Ui,Uf,Uo
        self.Ui=Variable(torch.rand(self.D_output,self.D_output),requires_grad=True)
        self.Uf=Variable(torch.rand(self.D_output,self.D_output),requires_grad=True)
        self.Uo=Variable(torch.rand(self.D_output,self.D_output),requires_grad=True)
        self.Uc=Variable(torch.rand(self.D_output,self.D_output),requires_grad=True)
    
    def train(self,x,c_last,h_last):
        # print(x)
        # print(h_last)
        i=self.sigmoid(x,h_last,'i')
        f=self.sigmoid(x,h_last,'f')
        o=self.sigmoid(x,h_last,'o')
        # print(i)
        # print(f.shape)
        c_hat=self.tanh(torch.matmul(self.Wc,x)+torch.matmul(self.Uc,h_last)+self.bc)
        # print(c_hat)
        c=torch.mul(f,c_last)+torch.mul(i,c_hat)
        h=torch.mul(o,self.tanh(c))

        return c,h

    def update(self):
        self.Wi.data -= alpha * self.Wi.grad.data 
        self.Wf.data -= alpha * self.Wf.grad.data 
        self.Wo.data -= alpha * self.Wo.grad.data 
        self.Wc.data -= alpha * self.Wc.grad.data 
        
        self.bi.data -= alpha * self.bi.grad.data 
        self.bf.data -= alpha * self.bf.grad.data 
        self.bo.data -= alpha * self.bo.grad.data 
        self.bc.data -= alpha * self.bc.grad.data 

        self.Ui.data -= alpha * self.Ui.grad.data 
        self.Uf.data -= alpha * self.Uf.grad.data 
        self.Uo.data -= alpha * self.Uo.grad.data 
        self.Uc.data -= alpha * self.Uc.grad.data 


    def tanh(self,T):
        T*=2
        max_ele=torch.max(T)
        T=T-max_ele 
        sigmoid=torch.div(torch.exp(T),torch.exp(T)+1)
        # print(sigmoid)
        return 2*sigmoid-1


    def sigmoid(self,x,h,flag):
        T=torch.rand(D_output)
        if flag=='i':
            T=torch.matmul(self.Wi,x)+torch.matmul(self.Ui,h)+self.bi
        elif flag=='f':
            T=torch.matmul(self.Wf,x)+torch.matmul(self.Uf,h)+self.bf            
        else:
            T=torch.matmul(self.Wo,x)+torch.matmul(self.Uo,h)+self.bo   
        max_ele=torch.max(T)
        T=T-max_ele 
        # prevent the exp to be too large
        return torch.div(torch.exp(T),torch.exp(T)+1)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# Supply functions


def softmax(X): #X=(d+1)*N
    max_ele=torch.max(X)
    X=X-max_ele # prevent the exp to be too large
    temp=torch.exp(X)
    divide=torch.sum(temp)
    return temp/divide

def cross_entropy(y_pred,y_tar):
    # y_pred[y_pred<=0.001]=0.001
    # y_pred[y_pred>=0.999]=0.999
    log_y_pred=torch.log(y_pred)
    log_y_1_pred=torch.log(1-y_pred)
    term=torch.mul(y_tar,log_y_pred)+torch.mul(1-y_tar,log_y_1_pred)
    # print(term)
    return -torch.sum(term)/D_output



            
class model:
    def __init__(self,D_input,D_output,D_batch,D_dict,eight_lines):
        self.D_input=D_input
        self.D_output=D_output
        self.D_batch=D_batch
        self.D_dict=D_dict
        self.eight_lines=eight_lines

        #the embedding layer, turn the word with one-hot representation to 
        raw_word_embedding=torch.rand(D_dict,D_input)
        leng=len(wordspool)
        for id in range(0,leng): # load the trained embedding to the vector
            if wordspool[id] in dictionary.keys():
                char=wordspool[id]
                raw_word_embedding[ dictionary[char] ]=torch.tensor(dict[char])
        raw_word_embedding=raw_word_embedding.t()
        self.WordEmbedding=Variable(raw_word_embedding,requires_grad=True) 
        
        #LSTM with the input a D_input size embedding, output is a D_output vec
        self.LSTM=MyLSTM(300,500,1)

        #the FFNN layer, turn the output of LSTM and remap into the dictionary
        self.FFNN=Variable(torch.rand(D_dict,D_output),requires_grad=True)


    def update(self):
        self.LSTM.update()
        self.WordEmbedding.data -= alpha * self.WordEmbedding.grad.data
        self.FFNN.data -= alpha * self.FFNN.grad.data


    def train(self,turns): # input is a collections of poems
        for poem in self.eight_lines:
            for i in range(0,turns): # for each poem, train turns times
                h_last=torch.rand(self.D_output)
                c_last=torch.rand(self.D_output)
                for ct in range(0,lines_of_poem*char_each_line-1): # for each word train
                    char=poem[ct] # pick out every char of the poem
                    one_hot=torch.zeros(self.D_dict)
                    one_hot[dictionary[char]]=1
                    x=torch.matmul(self.WordEmbedding,one_hot)

                    c , h = self.LSTM.train(x,c_last,h_last)
                    y_pred_raw = torch.matmul(self.FFNN,h)

                    y_pred = softmax(y_pred_raw)
                    char_tar=poem[ct+1]
                    y_tar  = torch.zeros(self.D_dict)
                    y_tar[dictionary[char_tar]] = 1
                    # print(y_pred)
                    loss=cross_entropy(y_pred,y_tar)
                    print(loss)
                    loss.backward(retain_graph=True)
                    self.update()
                    c_last=c
                    h_last=h


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# The main work is down here



def main():
    build_dict()
    eight_lines,D_dict=clr_poem()
    print(D_dict)
    training_model=model(D_input,D_output,D_batch,D_dict,eight_lines)
    training_model.train(turns)
    

main()
    