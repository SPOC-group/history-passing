import numpy as np
import itertools

def wolfram_number_to_binary(rule):
    binary=format(rule, "b")
    rule_binary=np.zeros(8, dtype='int')
    for i, num in enumerate(reversed(binary)):
        rule_binary[7-i]=num
    return rule_binary

def wolfram_to_new_notation(rule):
    binary=wolfram_number_to_binary(rule)
    new_notation=np.zeros(8, dtype='int')
    new_notation[0]=binary[7]
    new_notation[1]=binary[6]
    new_notation[2]=binary[3]
    new_notation[3]=binary[2]
    new_notation[4]=binary[5]
    new_notation[5]=binary[4]
    new_notation[6]=binary[1]
    new_notation[7]=binary[0]
    return new_notation

def binary_to_wolfram_notation(binary):
    rule=0
    for i in range(binary.size):
        rule+=binary[i]*2**(7-i)
    return rule

def new_notation_to_wolfram_notation(new_notation):
    binary=new_notation.copy()
    binary[0]=new_notation[7]
    binary[1]=new_notation[6]
    binary[2]=new_notation[3]
    binary[3]=new_notation[2]
    binary[4]=new_notation[5]
    binary[5]=new_notation[4]
    binary[6]=new_notation[1]
    binary[7]=new_notation[0]
    return binary_to_wolfram_notation(binary)
    
def binary_array(d=8):
    return np.array(list(itertools.product([0,1], repeat=d+1)), dtype='int')    

def game_of_life_rule(d=8, min_neighbours=2, max_neighbours=3, creation_neighbours=3):
    rule=np.zeros(2**(d+1), dtype='int')
    for i, conf in enumerate(binary_array(d)):
        alive_cells=np.sum(conf)
        if conf[0]==0:
            if alive_cells==creation_neighbours:
                rule[i]=1
        else:
            if alive_cells-1>=min_neighbours and alive_cells-1<=max_neighbours:
                rule[i]=1
    return rule

def totalistic_rule(array_rule, d=2):
    rule=np.zeros(2**(d+1), dtype='int')
    for i, conf in enumerate(binary_array(d)):
        alive_cells=np.sum(conf)
        if conf[0]==0:
            rule[i]=array_rule[alive_cells,0]
        elif conf[0]==1:
            rule[i]=array_rule[alive_cells-1, 1]
    return rule

def Marr_to_new_notation(notation):
    array_rule=np.zeros((3,2))
    for i in range(3):
        if notation[i]=='1':
            array_rule[i,0]=1
            array_rule[i,1]=1
        elif notation[i]=='+':
            array_rule[i,1]=1
        elif notation[i]=='-':
            array_rule[i,0]=1
        if notation[i]!='0' and notation[i]!='1' and notation[i]!='+' and notation[i]!='-':
            print('Error')
    return totalistic_rule(array_rule)

def Wolfram_to_Marr(rule):
    Marr=[]
    new_notation=wolfram_number_to_binary(rule)
    if new_notation[5]==0 and new_notation[7]==0:
        Marr.append('0')
    elif new_notation[5]==0 and new_notation[7]==1:
        Marr.append('-')
    elif new_notation[5]==1 and new_notation[7]==0:
        Marr.append('+')
    elif new_notation[5]==1 and new_notation[7]==1:
        Marr.append('1')
        
    if new_notation[1]==0 and new_notation[3]==0:
        Marr.append('0')
    elif new_notation[1]==0 and new_notation[3]==1:
        Marr.append('-')
    elif new_notation[1]==1 and new_notation[3]==0:
        Marr.append('+')
    elif new_notation[1]==1 and new_notation[3]==1:
        Marr.append('1')
        
    if new_notation[0]==0 and new_notation[2]==0:
        Marr.append('0')
    elif new_notation[0]==0 and new_notation[2]==1:
        Marr.append('-')
    elif new_notation[0]==1 and new_notation[2]==0:
        Marr.append('+')
    elif new_notation[0]==1 and new_notation[2]==1:
        Marr.append('1')
        
    return Marr[0]+Marr[1]+Marr[2]


def symmetric_rule(rule,d):
    symmetric_rule=list(reversed(rule))
    for i in range(d+1):
        if symmetric_rule[i]=='0':
            symmetric_rule[i]='1'
        elif symmetric_rule[i]=='1':
            symmetric_rule[i]='0'
    return symmetric_rule

def generate_independent_OT_rules(d=2):
    possibilities=list(itertools.product(['0','+', '-', '1'], repeat=d+1))
    possibilities=[list(rule) for rule in possibilities]
    independent_rules=[]
    for possibility in possibilities:
        if symmetric_rule(possibility,d) not in independent_rules:
            independent_rules.append(possibility)
    return independent_rules

        



#======================================================================================================================

def dual_rule(rule, d):
    symmetric_rule = list(reversed(rule))
    for i in range(d + 1):
        if symmetric_rule[i] == '0':
            symmetric_rule[i] = '1'
        elif symmetric_rule[i] == '1':
            symmetric_rule[i] = '0'
    return symmetric_rule


def generate_totalisitc_rules(d=2):
    possibilities = [''.join(r) for r in itertools.product(['0', '-', '+', '1'], repeat=d + 1)]
    return sorted(possibilities, key=lambda rule: to_number(rule))


def generate_totalistic_rules_unique(d=2):
    rules = generate_totalisitc_rules(d)
    independent_rules = []
    for rule in rules:
        if dual_rule(rule, d) not in independent_rules:
            independent_rules.append(rule)
    return independent_rules


def to_number(rule):
    rule_num = rule.replace('1', '3').replace('+', '1').replace('-', '2')
    return sum([4 ** int(power) * int(i) for power, i in enumerate(reversed(rule_num))])


def generate_generalized_threshold_rules(d=2):
    rules = generate_totalistic_rules_unique(d=d)
    tot = []
    for r in rules:
        for pattern in ['1+0', '1-0', '0+1', '0-1']:
            i = 0
            while i <= d and r[i] == pattern[0]:
                i += 1
            while i <= d and r[i] == pattern[1]:
                i += 1
            while i <= d and r[i] == pattern[2]:
                i += 1
            if i == d + 1:
                tot.append(r)
                break
    return tot


def generate_short_conjecture_rules(d=2):
    rules = generate_totalistic_rules_unique(d=d)
    tot = []
    for r in rules:
        i = 1  # any pattern for the first
        while i <= d and (r[i] == '+' or r[i] == '1'):
            i += 1
        if i == d + 1:
            tot.append(r)

    return tot
