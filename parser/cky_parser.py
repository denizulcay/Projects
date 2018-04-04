"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 2 - Parsing with Context Free Grammars 
Deniz Ulcay - du2147
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.{}\n".format(bps))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        parse_table = {}
        
        for i in range(len(tokens)):
            
            rng = (i, i + 1)
            token = tokens[i]
            token = (token,)
            print(token)
            
            if token in self.grammar.rhs_to_rules.keys():
                rules = self.grammar.rhs_to_rules[token]
            else:
                return False
            
            parse_table[rng] = [i[0] for i in rules]
            
        for length in range(2, len(tokens) + 1):
            
            for i in range(0, (len(tokens) - length) + 1):
                
                j = i + length
                parse_table[(i,j)] = []
                
                for k in range(i + 1, j):
                    
                    b = parse_table[(i,k)]
                    c = parse_table[(k,j)]
                    rhs_s = [rhs for rhs in list(itertools.product(b, c)) if rhs in self.grammar.rhs_to_rules.keys()]
                    
                    for rhs in rhs_s:
                        
                        rules = self.grammar.rhs_to_rules[rhs]
                        opts = [i[0] for i in rules]
                        parse_table[(i,j)] += opts
                                            
        if parse_table[(0, len(tokens))]:
            return True
        
        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        parse_table = {}
        probs = {}
        
        for i in range(len(tokens)):
            
            rng = (i, i + 1)
            token = tokens[i]
            token = (token,)
            print(token)
            
            parse_table[rng] = {}
            probs[rng] = {}
            
            if token in self.grammar.rhs_to_rules.keys():
                rules = self.grammar.rhs_to_rules[token]

                for j in rules:
                    parse_table[rng][j[0]] = j[1][0]
                    probs[rng][j[0]] = math.log(j[2])
                                
        for length in range(2, len(tokens) + 1):
            
            for i in range(0, (len(tokens) - length) + 1):
                
                j = i + length
                parse_table[(i,j)] = {}
                probs[(i,j)] = {}
                
                for k in range(i + 1, j):
                    
                    b = parse_table[(i,k)]
                    c = parse_table[(k,j)]
                    rhs_s = [rhs for rhs in list(itertools.product(b, c)) if rhs in self.grammar.rhs_to_rules.keys()]
                    
                    for rhs in rhs_s:
                        
                        rules = self.grammar.rhs_to_rules[rhs]
                        
                        for rule in rules:
                            
                            lhs = rule[0]
                            rhs = rule[1]
                            prob = rule[2]
                        
                            if lhs not in parse_table[(i,j)].keys():
                                
                                parse_table[(i,j)][lhs] = ((rhs[0],i,k), (rhs[1],k,j))
                                probs[(i,j)][lhs] = math.log(prob)
                            
                            else:
                                if probs[(i,j)][lhs] < math.log(prob):
                                    
                                    probs[(i,j)][lhs] = math.log(prob)
                                    parse_table[(i,j)][lhs] = ((rhs[0],i,k), (rhs[1],k,j))
                                    
        return parse_table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    
    if isinstance(chart[(i,j)][nt], str):
        return (nt, chart[(i,j)][nt])
    
    left = chart[(i,j)][nt][0]
    right = chart[(i,j)][nt][1]
    
    left_tree = get_tree(chart, left[1], left[2], left[0])
    right_tree = get_tree(chart, right[1], right[2], right[0])
    
    tree = (nt, left_tree, right_tree)
    
    return tree
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        
