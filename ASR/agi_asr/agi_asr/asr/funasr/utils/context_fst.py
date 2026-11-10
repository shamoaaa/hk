
from typing import Dict, List
import math

class ContextGraph:
    """ Context decoding graph, constructing graph using dict instead of WFST
        Args:
            context_list_path(str): context list path
            bpe_model(str): model for english bpe part
            context_score(float): context score for each token
    """
    def __init__(self,
                 hotword_list: List,
                 context_score: float = 6):
        self.context_score = context_score
        self.context_list = hotword_list
        self.graph = {0: {}}
        self.graph_size = 0
        self.state2token = {}
        self.back_score = {0: 0.0}
        self.build_graph(self.context_list)

    def build_graph(self, context_list: List[List[int]]):
        """ Constructing the context decoding graph, add arcs with negative
            scores returning to the starting state for each non-terminal tokens
            of hotwords, and add arcs with scores of 0 returning to the starting
            state for terminal tokens.
        """
        self.graph = {0: {}}
        self.graph_size = 0
        self.state2token = {}
        self.back_score = {0: 0.0}
        for context_token in context_list:
            now_state = 0
            for i in range(len(context_token)):
                if context_token[i] in self.graph[now_state]:
                    now_state = self.graph[now_state][context_token[i]]
                    if i == len(context_token) - 1:
                        self.back_score[now_state] = 0
                else:
                    self.graph_size += 1
                    self.graph[self.graph_size] = {}
                    self.graph[now_state][context_token[i]] = self.graph_size
                    now_state = self.graph_size
                    if i != len(context_token) - 1:
                        self.back_score[now_state] = -(i + 1) * self.context_score
                    else:
                        self.back_score[now_state] = 0
                    self.state2token[now_state] = context_token[i]

    def find_next_state(self,
                        now_state: int,
                        token: int):
        """ Search for an arc with the input being a token from the current state,
            returning the score on the arc and the state it points to. If there is
            no match, return to the starting state and perform an additional search
            from the starting state to avoid token consumption due to mismatches.
        """
        if token in self.graph[now_state]:
            return self.graph[now_state][token], self.context_score
        back_score = self.back_score[now_state]
        now_state = 0
        if token in self.graph[now_state]:
            return self.graph[now_state][token], back_score + self.context_score
        return 0, back_score
    
def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp