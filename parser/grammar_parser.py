"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 2 - Parsing with Context Free Grammars 
Deniz Ulcay - du2147
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """

        for lhs in self.lhs_to_rules:
            
            rule_list = self.lhs_to_rules[lhs]
            prob = fsum([i[2] for i in rule_list])
            
            if abs(1 - prob) > 0.00000000001:
                return False
            
            for rule in rule_list:
                
                if (len(rule[1]) != 2) and (len(rule[1]) != 1):
                    return False
                    
                elif len(rule[1]) == 2:
                    
                    for nonterm in rule[1]:
                        
                        if not nonterm.isupper():
                            return False
                
                else:
                    
                    term = rule[1][0]
                    
                    if term.isupper():
                        return False
            
        return True 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        valid = grammar.verify_grammar()
        
        if valid:
            sys.stderr.write("The grammar is a valid PCFG in CNF.")
        
        else:
            sys.stderr.write("ERROR: The grammar is NOT a valid PCFG in CNF.")
