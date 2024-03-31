#!/usr/bin/env python
import ptb
from ptb import all_rules , parse, traverse, generate_terminal_rules, remove_comma, remove_period, remove_quotation
import nltk
from nltk import CFG
from nltk import Tree
import pprint
import math
#import svgling
import time
import pickle
import glob
from tqdm import tqdm
#from CNF_hvalue_zero import un_CNF_hzero
#from h_twoCNF import unCNF_h_two
#from headnashiCNF import CNF_h_zero_headsame, un_CNF_hzero_headsame

# セル = {非終端記号: (対数確率, 導出に関する情報)} にビームを適用する
def apply_beam(cell, n_beam, p_beam):
    sorted_cell = sorted(cell.items(), key=lambda p: p[1][0], reverse=True)

    last = min(len(sorted_cell), n_beam)

    # 確率に基づくビームを適用
    if p_beam and len(sorted_cell) > 0:
        min_logp = sorted_cell[0][1][0] - p_beam
        while last > 0 and sorted_cell[last-1][1][0] < min_logp:
            last -= 1

    # 同点のものは削除しない
    while last < len(sorted_cell) and sorted_cell[last-1][1][0] == sorted_cell[last][1][0]:
        last += 1

    # 辞書にして返す
    # assert all(type(t) == tuple for t in sorted_cell[:last]), f"sorted_cell={sorted_cell}"
    beamed_cell = {symbol: logp for symbol, logp in sorted_cell[:last]}

    return beamed_cell

#セルに unary rule を1回適用した結果を返す
def apply_unary(cell, prob_grammar):
    unary_parent = {}
    for child_sym, child_value in cell.items():
        parents = prob_grammar.get(tuple([child_sym]), [])
        for parent_sym, rule_prob in parents:
            # p = math.log(rule_prob) + child_value[0]
            p = rule_prob + child_value[0]
            if parent_sym not in unary_parent or unary_parent[parent_sym][0] < p:
                unary_parent[parent_sym] = (p, ("unary", child_sym))
    return unary_parent

#セルに unary rule を2回適用し，結果を合併してビームを適用して返す
def apply_unary_twice(cell, prob_grammar, n_beam, p_beam, is_root=False):
    #１段目と２段目の保存場所は分けた方が混乱しない
    unary_parent = apply_unary(cell, prob_grammar)
    # 根のセル以外ではビームを適用
    if not is_root:
        unary_parent = apply_beam(unary_parent, n_beam, p_beam)

    #２段目の結果
    unary_grand_parent = apply_unary(unary_parent, prob_grammar)
    if not is_root:
        unary_grand_parent = apply_beam(unary_grand_parent, n_beam, p_beam)

    # 元のセル・１段目・２段目を合わせて一番よいものを選ぶ
    # Symbol も確率も同じなら導出が短い方を選ぶ．
    #       つまり確率が真に上回る場合のみ parent・grand_parent を使う　
    all_parent = cell # とりあえず孫のセルをセット

    for parent_sym, parent_logp in unary_parent.items():
        if parent_sym not in all_parent or all_parent[parent_sym][0] < parent_logp[0]:
            all_parent[parent_sym] = parent_logp

    for grand_parent_sym, grand_parent_logp in unary_grand_parent.items():
        if grand_parent_sym not in all_parent or all_parent[grand_parent_sym][0] < grand_parent_logp[0]:
            all_parent[grand_parent_sym] = grand_parent_logp

    # 根のセル以外ではビームを適用して返す
    if is_root:
        # 根のセルではビームを適用しない
        return all_parent
    else:
        return apply_beam(all_parent, n_beam, p_beam)

def pcky(text,prob_grammar,prob_lexicon, n_beam, p_beam, return_chart=False):#POSタグが36個なのでBeam >36
    n = len(text)
    # sentenceが単語のとき
    if n == 1:#トレーニングデータでは半分くらい(NP (NNP XXXX))であった
        parents = prob_lexicon.get(text[0], [])
        if parents  != []:
            parent = {}
            for parent_sym, rule_prob in parents:
                parent[parent_sym] = rule_prob
            return "(NP ({} {}))".format(max(parent, key = parent.get),text[0])
        elif parents == []:#辞書にルールがないとき
            if text[0].lower() in prob_lexicon: #小文字で入っているなら小文字と同じにする
                parents = prob_lexicon.get(text[i].lower(), []) 
                parent = {}
                for parent_sym, rule_prob in parents:
                    parent[parent_sym] = rule_prob
                    return "(NP ({} {}))".format(max(parent, key = parent.get),text[0])
            else:#小文字でもないなら固有名詞
                return "(NP (NNP {}))".format(text[0])
            
            

    ProbTable = [[{} for j in range(n+1)] for i in range(n+1)]
    # まずはlexicon
    for i in range(n):
        parents = prob_lexicon.get(text[i], []) 
        if parents != []:
            for parent_sym, rule_prob in parents:
                # p = math.log(rule_prob)
                p = rule_prob
                ProbTable[i][i+1][parent_sym] = (p,("lex",text[i]))
                
        else: #未知語ならunkのルールすべて追加
            parents = sorted(prob_lexicon.get('unk', []), reverse=False)
            for parent_sym, rule_prob in parents:
                # p = math.log(rule_prob)
                p = rule_prob
                ProbTable[i][i+1][parent_sym] = (rule_prob, ("lex",text[i]))

        ProbTable[i][i+1] = apply_unary_twice(ProbTable[i][i+1], prob_grammar, n_beam, p_beam, is_root=False)                      

    for l in range(2,n+1):
        for i in range(0, n-l+1):
            j = i + l
            for k in range(i+1,j):
                for left_sym, left_prob in ProbTable[i][k].items():
                    for right_sym, right_prob in ProbTable[k][j].items():
                        parents = prob_grammar.get((left_sym, right_sym), []) # 該当するルールがなければ [] を返す
                        for parent_sym, rule_prob in parents:
                            logp = rule_prob + left_prob[0] + right_prob[0]
                            if parent_sym not in ProbTable[i][j] or ProbTable[i][j][parent_sym][0] < logp:
                                ProbTable[i][j][parent_sym] = (logp, (k, left_sym, right_sym))

            # binary rule の結果に一旦 beam を適用
            is_root = (i == 0 and j == n)
            #ProbTable[i][j] = apply_beam(ProbTable[i][j], n_beam, p_beam)

            # mtzk: uanry rule を最大2回適用
            ProbTable[i][j] = apply_unary_twice(ProbTable[i][j], prob_grammar, n_beam, p_beam, is_root)
                
    # ProbTable に辞書の形で全て情報が入っているので
    #       リストに変換しなくても木はすぐ取り出せる
    #return ProbTable, text
    if return_chart:
        return ProbTable
    else:
        return chart_to_tree("ROOT", 0, n, ProbTable, text)
   

# chart_to_tree(sym, i, j, ProbTable, text) 
#   = text 中の [i, j) までの句について sym から導出される最適な部分木
def chart_to_tree(sym, i, j, ProbTable, text):
    if sym in ProbTable[i][j]:
        link = ProbTable[i][j][sym]
        if link[1][0] == "lex":
            return Tree(sym, [link[1][1]]) # e.g., (VBZ 'runs')
        elif link[1][0] == "unary":
            return Tree(sym, [chart_to_tree(link[1][1], i, j, ProbTable, text)]) # e.g., (NP (NNP ...))
        elif type(link[1][0]) == int: # binary rule
            k, left_sym, right_sym = link[1]
            return Tree(sym, [chart_to_tree(left_sym, i, k, ProbTable, text),
                              chart_to_tree(right_sym, k, j, ProbTable, text)])
        else:
            assert False, (f"unexpected ProbTable cell for (i, j)=({i}, {j}), sym={sym}: {ProbTable[i][j][sym]}\n" +
                           f"in \"{' '.join(text)}\"")
    else:
        #「リンク切れ」：しょうがないので (sym (X 単語i) ... (X 単語j-1)) を返す
        if sym == "ROOT" and i == 0 and j + 1 == len(ProbTable):
            # 文全体に対して ROOT が導出できなかった -> ありうるケース
            return Tree("ROOT", [Tree("X", [w]) for w in text])
        else:
            # ProbTable の中の「リンク」が切れてる -> バグ!
            assert False, (f"no derivations from {sym} found for\n" +
                           ' '.join(text[i:j]) + "\n" +
                           f"(i={i}, j={j}) in\n" +
                           ' '.join(text))
    
def make_tree(BP):
    n = len(BP) - 1
    if len(BP[0][n]) == 2:#最後のセルがunaryかどうか
        root = BP[0][n][1] 
        tree = Tree(root[0] , [])
        unaryHead = root[1][1]
        unaryNode = Tree(unaryHead, [])
        tree[0:] = [unaryNode]
        
        root = BP[0][n][0] 
        leftHead = root[1][1]
        leftNode = Tree(root[1][1], [])
        rightHead = root[1][2]
        rightNode = Tree(root[1][2], [])
        tree[0][0:] = [leftNode, rightNode]
    else:
        root = BP[0][n][0] 
        tree = Tree(root[0] , [])
        leftHead = root[1][1]
        leftNode = Tree(root[1][1], [])
        rightHead = root[1][2]
        rightNode = Tree(root[1][2], [])
        tree[0:] = [leftNode, rightNode]    
    i = 0
    k = BP[0][n][0][1][0]
    j = n
    nodeList = [(tree, [i,k,j])]
    while nodeList != []:
        node, ikj = nodeList.pop()  
        i = ikj[0]
        k = ikj[1]
        j = ikj[2]
        if len(node) == 2:
            if isinstance(node[0], Tree): # 左側
                for left_bp in BP[i][k]:
                    if left_bp[0] == node[0].label():
                        if left_bp[1][0] == 'lex':
                            node[0][0:] = [left_bp[1][1]]

                        elif left_bp[1][0] == 'unary':
                            unaryHead = left_bp[1][1]
                            unaryNode = Tree(unaryHead, [])
                            node[0][0:] = [unaryNode]
                            nodeList.append((node[0], [i,k,k]))

                        else:
                            leftHead = left_bp[1][1]
                            leftNode = Tree(leftHead, [])
                            rightHead = left_bp[1][2]
                            rightNode = Tree(rightHead, [])
                            node[0][0:] = [leftNode, rightNode]

                            nodeList.append((node[0], [i,left_bp[1][0],k]))

            if isinstance(node[1], Tree): # 右側
                for right_bp in BP[k][j]:
                    if right_bp[0] == node[1].label():
                        if right_bp[1][0] == 'lex':
                            node[1][0:] = [right_bp[1][1]]

                        elif right_bp[1][0] == 'unary':
                            unaryHead = right_bp[1][1]
                            unaryNode = Tree(unaryHead, [])
                            node[1][0:] = [unaryNode]
                            nodeList.append((node[1], [k,k,j]))

                        else:
                            leftHead = right_bp[1][1]
                            leftNode = Tree(leftHead, [])
                            rightHead = right_bp[1][2]
                            rightNode = Tree(rightHead, [])
                            node[1][0:] = [leftNode, rightNode]

                            nodeList.append((node[1], [k,right_bp[1][0],j]))
        
        elif len(node) == 1:# unary
            for unary_bp in BP[i][j]:
                if unary_bp[0] == node[0].label() and unary_bp[1][0] == 'lex':
                    node[0][0:] = [unary_bp[1][1]]
                elif unary_bp[0] == node[0].label() and unary_bp[1][0] != 'unary':
                    leftHead = unary_bp[1][1]
                    leftNode = Tree(leftHead, [])
                    rightHead = unary_bp[1][2]
                    rightNode = Tree(rightHead, [])
                    node[0][0:] = [leftNode, rightNode]

                    nodeList.append((node[0], [i,unary_bp[1][0],j]))
                
                elif unary_bp[0] == node[0].label() and unary_bp[1][0] == 'unary':
                    unaryHead = unary_bp[1][1]
                    unaryNode = Tree(unaryHead, [])
                    node[0][0:] = [unaryNode]
                    nodeList.append((node[0], [i,k,j]))

    ROOT = Tree('ROOT',[])
    ROOT[0:] = [tree]
    return ROOT

def unCNF(tree, expandUnary=True,childChar="|", parentChar="^", unaryChar="+"):
    nodeList = [(tree, [])]
    c = 0
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):
            # TORIAEZU: 先に PRT|ADVP を PRT に直すべき
            childIndex = node.label().find(childChar + "<")
            #childIndex = node.label().find(childChar)
            if childIndex != -1 and node.label().find('_') != -1:
                nodeIndex = parent.index(node)
                parent.remove(parent[nodeIndex])
                # Generated node was on the left if the nodeIndex is 0 which
                # means the grammar was left factored.  We must insert the children
                # at the beginning of the parent's children
                
                if nodeIndex == 0:
                    parent.insert(0, node[0])
                    parent.insert(1, node[1])
                    
                else:
                    parent.insert(nodeIndex, node[0])
                    parent.insert(nodeIndex + 1, node[1])
                    #parent.extend([node[0], node[1]])

                # parent is now the current node so the children of parent will be added to the agenda
                node = parent
            elif childIndex != -1 and node.label().find('_') == -1:#headのところ
                nodeIndex = parent.index(node)
                parent.remove(parent[nodeIndex])
                parent.insert(nodeIndex, node[0])
                node = parent
                
            else:
                parentIndex = node.label().find(parentChar)
                if parentIndex != -1:
                    # strip the node name of the parent annotation
                    node.set_label(node.label()[:parentIndex])
                    
                # expand collapsed unary productions
                if expandUnary == True:
                    unaryIndex = node.label().find(unaryChar)
                    if unaryIndex != -1:
                        newNode = Tree(
                            node.label()[unaryIndex + 1 :], [i for i in node]
                        )
                        node.set_label(node.label()[:unaryIndex])
                        node[0:] = [newNode]

            for child in node:
                nodeList.append((child, node))
    return tree

def make_text(tree):
    state = []
    def pre(tree, state):
        if tree.leaf():
            return state + [tree.leaf().word]
        return state
    return traverse(tree, pre, state=[])

#-------------
# main
#-------------
import sys

def main():
    
    if len(sys.argv) < 3:
        print(f"{sys.argv[0]} <grammar> <input> [n_beam p_beam]", file=sys.stderr)
        sys.exit(1)
        
    grammar = sys.argv[1]
    fn_input = sys.argv[2]
    n_beam = 150
    p_beam = math.log(10000) # トップの 1/10000 まで残す
    if len(sys.argv) == 5:
        n_beam = int(sys.argv[3])
        if sys.argv[4] != "None":
            p_beam = math.log(float(sys.argv[4]))
        else:
            p_beam = None
        
    with open(f'{grammar}_ter.pickle', mode='br') as fi:
        prob_lexicon = pickle.load(fi)
    with open(f'{grammar}_non.pickle', mode='br') as fi:
        prob_grammar = pickle.load(fi)

    # 文法を得た時点でやっておくべき
    for k, ps in prob_lexicon.items():
        prob_lexicon[k] = [(sym, math.log(p)) for sym, p in ps]
    for k, ps in prob_grammar.items():
        prob_grammar[k] = [(sym, math.log(p)) for sym, p in ps]
        
    
    with open(fn_input) as f:
        s_parsed = parse(f.read())
    
    text_list = []
    for i in s_parsed:
        ptb.add_root(i)
        ptb.remove_empty_elements(i)
        ptb.simplify_labels(i)
        text = make_text(i)
        text_list.append(text)
        
    for text in tqdm(text_list):
        #ROOT の導出に失敗するケースも pcky にまかせる
        s = pcky(text, prob_grammar, prob_lexicon, n_beam, p_beam)
        s = un_CNF_hzero(s)
        #s = un_CNF_hzero_headsame(s)
        #s = unCNF(s)
        #s = unCNF_h_two(s)
        #s.un_chomsky_normal_form()
        s = next(parse(str(s)))
        print(str(s))


if __name__ == "__main__":
    main()
