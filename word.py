import json
import jieba
from langconv import *
import random

D_vec=300
dict={}
wordspool=[]
dictionary={}

ct=0
def build_dict():
    with open("embedding.word","r",encoding='UTF-8') as f:
        f.readline()
        while 1:
            line=f.readline()
            if not line:
                break
            split=line.split()
            char=split[0]
            # print(char)
            vec=[0]*D_vec  #fot the dimension of the vector is 300
            for i in range(0,D_vec):
                vec[i]=split[i+1]
            dict[char]=vec
            wordspool.append(char)
        f.close

def clr_poem(addr="test.json"):
    eight_lines=[]
    with open(addr,"r",encoding='UTF-8') as f:
        all = json.loads(f.read())
        for poem in all:
            content=poem['paragraphs']
            if len(content)==4:
                if len(content[0])==12:
                    print(poem['title'])
                    Apoem=[]
                    for line in content:    
                        line = Converter('zh-hans').convert(line)
                        print(line)
                        for j in range(0,12):
                            in_char=line[j]
                            if line[j]=='，' or line[j]=='。':
                                in_char=','
                            try:
                                if dictionary.has_key(in_char):
                                    dictionary[in_char]=ct
                                    ct+=1
                                    print(ct)
                                Apoem.append(dict[in_char])
                            except:
                                ran=random.randint(0,1000)
                                in_char=wordspool[ran]
                                Apoem.append(dict[in_char])
                    eight_lines.append(Apoem)
    print(len(eight_lines))
    return eight_lines



# eight_lines=clr_poem()

# print(len(eight_lines))



# build_dict()
# clr_poem()


# About the simply work: https://blog.csdn.net/u012052268/article/details/77823970