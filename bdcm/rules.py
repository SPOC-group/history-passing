
import numpy as np
import itertools


def dual_rule(rule, d):
    symmetric_rule = list(reversed(rule))
    for i in range(d + 1):
        if symmetric_rule[i] == '0':
            symmetric_rule[i] = '1'
        elif symmetric_rule[i] == '1':
            symmetric_rule[i] = '0'
    return ''.join(symmetric_rule)


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
    rule_num = rule.replace('1', '2').replace('0','1').replace('+', '0').replace('-', '3')
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
        i = 0  # any pattern for the first
        while i <= d and (r[i] == '+' or r[i] == '0'):
            i += 1
        if i == d:
            tot.append(r)

    return tot

def generate_short_conjecture_rules_unresolved(d=2):
    conj_rules = generate_short_conjecture_rules(d=d)
    gen_rules = generate_generalized_threshold_rules(d=d)
    return [r for r in conj_rules if (r not in gen_rules and ('1' in r or '-' in r))]

def generate_majority_threshold_rules(d=2):
    """
    This is the set of rules used in the paper.
    """
    rules = generate_totalistic_rules_unique(d=d)
    tot = []
    for r in rules:
        for pattern in ['0+1', '0-1']:
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

X_POS = 0
Y_POS = 1

DEAD = 0
ALIFE = 1


class TotalisiticRule:
    """

    rule = '++010-'

    When we have i alife neighbours, then given our
    rule[i] =
    + : cell stays at the same state
    - : cell switches state
    0 : cell becomes DEAD
    1 : cell becomes ALIFE
    """

    def __init__(self, rule, d):
        self.d = d
        self.rule = rule
        self.allowed_alive = np.zeros(self.d + 1).astype(bool)
        self.allowed_dead = np.ones(self.d + 1).astype(bool)

        if len(rule) != d + 1:
            raise ValueError(f"Rule needs to have length {d + 1}, but is {len(rule)} long.")

        for i, symb in enumerate(rule):
            if symb == '+':
                self.allowed_alive[i] = True
                self.allowed_dead[i] = True
            elif symb == '-':
                self.allowed_alive[i] = False
                self.allowed_dead[i] = False
            elif symb == '0':
                self.allowed_alive[i] = False
                self.allowed_dead[i] = True
            elif symb == '1':
                self.allowed_alive[i] = True
                self.allowed_dead[i] = False
            else:
                raise ValueError(f"Symbol {symb} in rule not supported! Choose between '0,1,+,-'!")

        N_ALIFE = np.arange(self.d + 1)

        self.allowed_alife_neighbours = {
            DEAD: N_ALIFE[self.allowed_dead],
            ALIFE: N_ALIFE[self.allowed_alive]
        }

    def __str__(self):
        return f'game=Tot.{self.rule}'

    @property
    def name(self):
        return self.rule

    @staticmethod
    def symmetric_rule(rule, d):
        symmetric_rule = rule.copy()[::-1]
        for i in range(d + 1):
            if symmetric_rule[i] == '0':
                symmetric_rule[i] = '1'
            elif symmetric_rule[i] == '1':
                symmetric_rule[i] = '0'
        return symmetric_rule

    @staticmethod
    def generate_independent_rules(d=2):
        possibilities = list(itertools.product(['0', '1', '+', '-'], repeat=d + 1))
        possibilities = [list(rule) for rule in possibilities]
        independent_rules = []
        for possibility in possibilities:
            if TotalisiticRule.symmetric_rule(possibility, d) not in independent_rules:
                independent_rules.append(possibility)
        return [''.join(r) for r in independent_rules]

