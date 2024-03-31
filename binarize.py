#!/usr/bin/env python
import math
from collections import Counter
import ptb
from ptb import all_rules, parse, traverse, generate_terminal_rules, remove_comma, remove_period
import nltk
from nltk import CFG
from nltk import Tree
import time
import glob
import pickle
import sys
from tqdm import tqdm
#from CNF_hvalue_zero import CNF_h_zero, un_CNF_hzero
#from h_twoCNF import CNF_h_two, unCNF_h_two
#from headnashiCNF import CNF_h_zero_headsame, un_CNF_hzero_headsame
POSTags = ['WRB','DT','CC','RB','RBR','PRP','EX','WDT','TO','JJS','CD','JJ','VBP','PRP$','WP','RBS','NN','VBD','RP','SYM','POS','PDT','IN','NNP','MD','JJR','WP$','FW','UH','NNS','VBN','VBZ','VBG','NNPS','VB','LS']

with open('head_table.pickle', mode='br') as fi:
    head_table = pickle.load(fi)
    
def search_head(parent, table, children):
    if table.get(parent,[]) != []: #NPもしくはそれ以外でないなら
        if table.get(parent,[])[0] == 'left':
            for sym in table.get(parent,[])[1]:
                for index,child in enumerate(children):
                    if sym == child:
                        return index,child
            return 0,children[0]
        elif table.get(parent,[])[0] == 'right':
            for sym in table.get(parent,[])[1]:
                for index,child in enumerate(reversed(children)):
                    if sym == child:
                        return len(children)-index-1,child
            return len(children)-1,children[len(children)-1]
    elif parent == 'NP':
        # 1 last-wordがposタグなら
        if children[len(children)-1] in POSTags:
            return len(children)-1, children[len(children)-1]
        # 2
        for sym in ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'POS', 'JJR']:
            for index,child in enumerate(reversed(children)):
                if sym == child:
                    return len(children)-index-1,child
        # 3
        for index,child in enumerate(children):
            if child == 'NP':
                return index,child
        # 4
        for sym in ['$', 'ADJP', 'PRN']:
            for index,child in enumerate(reversed(children)):
                if sym == child:
                    return len(children)-index-1,child
        # 5
        for index,child in enumerate(reversed(children)):
            if child == 'CD':
                return len(children)-index-1,child
        # 6
        for sym in ['JJ', 'JJS', 'RB','QP']:
            for index,child in enumerate(reversed(children)):
                if sym == child:
                    return len(children)-index-1,child
        # 7
        return len(children)-1,children[len(children)-1]
    
    else:
        return 0,children[0]
    
def tag_PA(tx):
    def proc(tx, st):
        if tx.leaf():
            tx.leaf().pos = tx.leaf().pos + '^' + '<' + tx.parent().symbol().label + '>'
    traverse(tx, proc)

def CNF(tree, head_table, horzMarkov=1, vertMarkov=0, childChar="|", parentChar="^"): # paperと同じ形 binaryもunaryつけた
    nodeList = [(tree, [tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):
            
            # parent annotation
            parentString = ""
            originalNode = node.label()
            if vertMarkov != 0 and node != tree and isinstance(node[0], Tree):
                parentString = "{}<{}>".format(parentChar, "-".join(parent))
                node.set_label(node.label() + parentString)
                parent = [originalNode] + parent[: vertMarkov - 1]
                
            for child in node:
                nodeList.append((child, parent))
                
            # binaryにもunaryをつけてみる
            if node.label().find(childChar) == -1 and len(node) == 2:
                childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = []  # delete the children
                curNode = node
                numChildren = len(nodeCopy)

                head_index,head = search_head(originalNode, head_table, childNodes)
                if head_index == 0:
                    newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[1],parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [newNode]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]
                    
                    # 一番下のunary    
                    newHead = "{}{}<[{}]>{}".format(originalNode,childChar,head,parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [newNode, nodeCopy.pop()]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]
                else:
                    newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[0],head,parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [newNode]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]    
                    
                    # 一番下のunary   
                    newHead = "{}{}<[{}]>{}".format(originalNode,childChar,head,parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [nodeCopy.pop(0), newNode]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]
                
            # chomsky normal form factorization
            if len(node) > 2:
                childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = []  # delete the children
                curNode = node
                numChildren = len(nodeCopy)
                
                head_index, head = search_head(originalNode, head_table, childNodes)
                
                #cordinated phrases
                if head_index > 1 and childNodes[head_index-1] == 'CC':
                    head = childNodes[head_index-2]
                    head_index = head_index-2
                
                if head_index == numChildren-1:#headが一番右なら右下がり
                    for i in range(1, numChildren - 1):
                        if i == 1:
                            newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[0],head,parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                            
                            newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[i],head,parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [nodeCopy.pop(0), newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                        else:
                            newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[i],head,parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [nodeCopy.pop(0), newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]

                    # 一番下のunary    
                    newHead = "{}{}<[{}]>{}".format(originalNode,childChar,head,parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [nodeCopy.pop(0), newNode]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]
                        
                elif head_index == 0:#headが一番左なら左下がり
                    for i in range(1, numChildren - 1):
                        if i == 1:
                            newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[numChildren - 1],parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                            
                            newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[-i-1],parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [newNode,nodeCopy.pop()]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                        else:
                            newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[numChildren - i-1],parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [newNode, nodeCopy.pop()]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                    # 一番下のunary    
                    newHead = "{}{}<[{}]>{}".format(originalNode,childChar,head,parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [newNode, nodeCopy.pop()]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]
                    
                    
                else:
                    # まずはheadより左の子を右下がりで二分木化 
                    for i in range(1,head_index+1):
                        if i == 1:
                            newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[0],head,parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                            
                            if head_index == 1:
                                newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[numChildren-1],parentString)
                                newNode = Tree(newHead, [])
                                curNode[0:] = [nodeCopy.pop(0), newNode]
                                curNode = newNode
                                curNode[0:] = [child for child in nodeCopy]
                                
                            else:
                                newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[i],head,parentString)
                                newNode = Tree(newHead, [])
                                curNode[0:] = [nodeCopy.pop(0), newNode]
                                curNode = newNode
                                curNode[0:] = [child for child in nodeCopy]
                            
                        elif i == head_index:
                            newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[numChildren-1],parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [nodeCopy.pop(0), newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                        else:
                            #TMP
                            newHead = "{}{}<{}_[{}]>{}".format(originalNode,childChar,childNodes[i],head,parentString)
                            newNode = Tree(newHead, [])
                            curNode[0:] = [nodeCopy.pop(0), newNode]
                            curNode = newNode
                            curNode[0:] = [child for child in nodeCopy]
                    
                    # headを含む残りを左下がりでを二分木化
                    for i in range(1,numChildren-head_index-1):
                        newHead = "{}{}<[{}]_{}>{}".format(originalNode,childChar,head,childNodes[-i-1],parentString)
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]
                        curNode = newNode
                        curNode[0:] = [child for child in nodeCopy]
                    
                    # 一番下のunary    
                    newHead = "{}{}<[{}]>{}".format(originalNode,childChar,head,parentString)
                    newNode = Tree(newHead, [])
                    curNode[0:] = [newNode, nodeCopy.pop()]
                    curNode = newNode
                    curNode[0:] = [child for child in nodeCopy]
                        
    return tree
    
if __name__ == "__main__":
    path = glob.glob('/home/corpus/PTB3/treebank_3/parsed/mrg/wsj/*') 
    # トレーニングセットは02~21
    path = path[2:22]
    path_list = []
    for i in path:
        path_list.append(glob.glob('{}/wsj_*'.format(i)) )

    s = ""
    for XX in path_list:
        for path in XX:
            f = open(path)
            mark = f.read()
            s += mark
            f.close()
    #sは全ての構文木のstring

    treebank = ""
    s_parsed = parse(s)
    # チョムスキー標準形に変換
    print("ptb -> CNF")
    for i in tqdm(s_parsed):
        ptb.remove_empty_elements(i)
        ptb.simplify_labels(i)
        ptb.add_root(i)
        #tag_PA(i)
        #parent_annotation(i)
        i = nltk.Tree.fromstring(str(i))
        #i = CNF_h_zero_headsame(i, head_table, vertMarkov=0)
        #i = CNF_h_two(i, head_table, vertMarkov=2)
        #i = CNF_h_zero(i, head_table, vertMarkov=2)
        #i = CNF(i, head_table, vertMarkov=1)
        i.chomsky_normal_form(horzMarkov =0,vertMarkov=0)
        treebank += str(i)

    rulelist = []
    rulelist_terminal = []
    parsed_tree = parse(treebank)
    # すべてのルールの重複ありの非終端記号ルールリストを作成
    # すべてのルールの重複ありの終端記号ルールリストを作成
    print("collecting nonterm & lex rules")
    for i in tqdm(parsed_tree):
        rulelist += all_rules(i)
        rulelist_terminal += generate_terminal_rules(i)        

    #重複あり nonterm rules 
    rulelist_split = [rule.split(' -> ') for rule in rulelist]
    # [(親, (子供1,..))]
    rulelist_split = [(rule[0],tuple(rule[1].split(' '))) for rule in rulelist_split]

    #重複あり lex rules
    # [(POS, 単語)]
    rulelist_terminal_split = [rule.split(' -> ') for rule in rulelist_terminal]


    #重複なし nonterm rules [(親, (子供1,..))]
    nonterminal_rules = set(rulelist)
    text_nonterminal_split = [rule.split(' -> ') for rule in nonterminal_rules]
    text_nonterminal_split = [(rule[0], tuple(rule[1].split(' '))) for rule in text_nonterminal_split]

    #重複なし lex rules [(POS, 単語)]
    terminal_rules = set(rulelist_terminal)
    text_terminal_split = [rule.split(' -> ') for rule in terminal_rules]

    nonterminal_left = [rule[0] for rule in rulelist_split]
    terminal_left = [rule[0] for rule in rulelist_terminal_split]

    print("counting rules and symbols")
    counter_rulelist = Counter(rulelist)
    counter_non_left = Counter(nonterminal_left)
    counter_terminal = Counter(rulelist_terminal)
    counter_ter_left = Counter(terminal_left)
    # unk はトータルの出現回数で判定するので単語のカウントも必要
    counter_ter_right = Counter(rule[1] for rule in rulelist_terminal_split)

    
    unk_num = 3
    time_sta = time.time()

    print("prob estim for nonterm rules", file=sys.stderr)
    nonterminal_dict = {}
    # children = set([rule[1] for rule in text_nonterminal_split])
    for parent, children in text_nonterminal_split:
        A_count = counter_non_left[parent]
        str_rule = parent + ' -> ' + ' '.join(children)
        ABC_count = counter_rulelist[str_rule]
        p = ABC_count / A_count
        if children in nonterminal_dict:
            nonterminal_dict[children].append((parent, p))
        else:
            nonterminal_dict[children] = [(parent, p)]

    print("prob estim for lex rules", file=sys.stderr)
    terminal_dict = {}

    # UNK はトータルの出現回数が閾値以下のもの
    unk_pos_count = Counter()
    for pos, word in text_terminal_split:
        if counter_ter_right[word] <= unk_num:
            # pos が UNK の品詞として出現した回数をカウント
            unk_pos_count[pos] += counter_ter_right[word]
        else:
            rule = f"{pos} -> {word}"
            ABC_count = counter_terminal[rule]
            A_count = counter_ter_left[pos]
            p = ABC_count / A_count
            if word not in terminal_dict:
                terminal_dict[word] = []
            terminal_dict[word].append((pos, p))

    # 品詞 -> UNK のルールを作る
    terminal_dict["unk"] = [(pos, count / counter_ter_left[pos]) for pos, count in unk_pos_count.items()]

    time_end = time.time()
    tim = time_end- time_sta
    print(tim)

    import pickle
    with open('h0v0_nltk_non.pickle', mode='wb') as fo:
        pickle.dump(nonterminal_dict, fo)
    with open('h0v0_nltk_ter.pickle', mode='wb') as fo:
        pickle.dump(terminal_dict, fo)
