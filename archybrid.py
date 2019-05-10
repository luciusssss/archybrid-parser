#coding=utf-8

# reference: https://github.com/MathijsMul/lm_parser/blob/master/arc_hybrid.py

# Template of modified arc-hybrid systems
# Add 'SWAP' action for non-projective trees

import pprint

global SHIFT, LEFT_ARC, RIGHT_ARC, SWAP 
SHIFT, LEFT_ARC, RIGHT_ARC, SWAP = 0, 1, 2, 3

class Stack:
    """
    Class for stack in arc hybrid configuration system.
    Attributes:
        contents: words currently in stack
    """

    def __init__(self):
        self.contents = []

    def __getitem__(self, item):
        if abs(item) <= len(self):
            return self.contents[item]

    def __len__(self):
        return len(self.contents)

    def reduce(self):
        x = self.contents[-1]
        self.contents.pop()
        return x

    def add(self, word):
        self.contents.append(word)

class Buffer:
    """
    Class for buffer in arc hybrid configuration system.
    Attributes:
        contents: words currently in buffer
    """

    def __init__(self, sentence):
        self.contents = [wordid for wordid in sentence]
        # '0' indicates 'ROOT'
        self.contents.append(0) 

    def __getitem__(self, item):
        if item < len(self):
            return self.contents[item]

    def __len__(self):
        return len(self.contents)

    def shift(self):
        x = self.contents[0]
        self.contents.pop(0)
        return x
    
    def insert_after_top(self, item):
        self.contents.insert(1, item)

class Arcs:
    """
    Class for arcs in arc hybrid configuration system.
    Attributes:
        contents: arcs inferred up until present moment, formatted as (head, modifier, label)
    """

    def __init__(self):
        self.contents = []

    def __repr__(self):
        return(str(self.contents))

    def add(self, arc):
        """
        Add arc
        :param arc: (head, modifier, label)
        :return:
        """

        self.contents.append(arc)

    def load(self, arcs_given):
        self.contents = arcs_given

    def unlabeled_arcs(self):
        return(list(map(lambda triple: (triple[0], triple[1]), self.contents)))

    def contains(self, head, dependent):
        # Check if (head, dependent, label) in arcs for any label

        unlabeled_arcs = self.unlabeled_arcs()
        return((head, dependent) in unlabeled_arcs)

    def child_still_has_children(self, child):
        # Check if word has no dependents of its own before being reduced

        unlabeled_arcs = self.unlabeled_arcs()
        (parents, children) = zip(*unlabeled_arcs)
        return(child in parents)

    def get_label(self, head, dependent):
        # Get label of included arc

        index_arc = self.unlabeled_arcs().index((head, dependent))
        label = self.contents[index_arc][2]
        return(label)

class Configuration:
    """
    Total arc hybrid configuration.
    Attributes:
        stack: Stack object
        buffer: Buffer object
        arcs: Arcs object
        contents: dictionary of the above
    """

    def __init__(self, sentence, sufficient_info=True):
        self.stack = Stack()
        self.buffer = Buffer(list(range(1, len(sentence)+1)))
        self.arcs = Arcs()
        self.contents = {'stack': self.stack.contents,
                         'buffer': self.buffer.contents,
                         'arcs': self.arcs.contents}
        
        if sufficient_info:
            # Get fathers and rdeps
            self.father = list(range(len(sentence)+1))
            self.rdeps = list(range(len(sentence)+1))
            self.left_nodes = []
            self.right_nodes = []
            for i in range(len(sentence)+1):
                self.left_nodes.append([])
                self.right_nodes.append([])

            for i in range(len(sentence)+1):
                self.rdeps[i] = set([])

            for word in sentence:
                child = int(word['id'])
                fa = int(word['father'])
                self.father[child] = fa
                self.rdeps[fa].add(child)
                if child < fa:
                    (self.left_nodes[fa]).append(child)
                elif child > fa:
                    (self.right_nodes[fa]).append(child)

            #############################
            # Calculate projective order
            self.proj_order = list(range(len(sentence)+1))
            self.porj_order_cnt = 0
            self.calculate_proj_order(0)
            #############################

        

    def pretty_print(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.contents)

    def shift(self):
        """
        Top word of buffer shifted to stack.
        :return:
        """
        self.stack.add(self.buffer[0])
        self.buffer.shift()

    def swap(self):
        """
        Insert top item of stack after top word of buffer
        """
        stack_top = self.stack[-1]
        self.stack.reduce()
        self.buffer.insert_after_top(stack_top)
        

    def left_arc(self, label):
        """
        Removes top item of stack, and attaches it 
        as a modifier to top word of buffer 
        :return:
        """
        self.arcs.add((self.buffer[0], self.stack[-1], label))
        self.stack.reduce()

    def right_arc(self, label):
        """
        Top word of stack depends on second word of stack.
        :return:
        """
        self.arcs.add((self.stack[-2], self.stack[-1], label))
        self.stack.reduce()

    def oracle_update(self, transiton, shift_case):
        # Apply updates according to Figure 1 
        # in "Arc-Hybrid Non-Projective Dependency Parsing 
        # with a Static-Dynamic Oracle"
        s0 = self.stack[-1] if len(self.stack) > 0 else None
        b0 = self.buffer[0] if len(self.buffer) > 0 else None
        if transiton == SHIFT:
            if shift_case == 2:
                if self.father[b0] in self.stack.contents[:-1] and b0 in self.rdeps[self.father[b0]]:
                    self.rdeps[self.father[b0]].remove(b0)
                    blocked_dps = [d for d in self.rdeps[b0] if d in self.stack]
                    for d in blocked_dps:
                        self.rdeps[b0].remove(d)
        elif transiton == LEFT_ARC or transiton == RIGHT_ARC:
            self.rdeps[s0] = set([])
            if s0 in self.rdeps[self.father[s0]]:
                self.rdeps[self.father[s0]].remove(s0)

    def apply_transition(self, transition):
        if transition == SHIFT:
            self.shift()
        elif transition == SWAP:
            self.swap()
        elif transition[0] == LEFT_ARC:
            self.left_arc(transition[1])
        elif transition[0] == RIGHT_ARC:
            self.right_arc(transition[1])

    def extract_features(self, feature_dict):
        buffer_features = [self.buffer[idx] for idx in feature_dict['buffer']]
        stack_features = [self.stack[idx] for idx in feature_dict['stack']]
        return(stack_features + buffer_features)

    def transition_admissible(self, transition):
        """
        Check if transition is admissible.
        :param transition: candidate transition
        :return: boolean
        """

        if transition == SHIFT:
            return(len(self.buffer) > 0) and self.buffer[0] != 0
        elif transition == SWAP:
            return (len(self.buffer) > 1 and len(self.stack) > 0 and self.stack[-1] < self.buffer[0])
        elif transition == LEFT_ARC:
            return(len(self.buffer) > 0 and len(self.stack) > 0 and self.stack[-1] != 0)
        elif transition == RIGHT_ARC:
            return(len(self.stack) > 1 and self.stack[-1] != 0)

    def is_empty(self):
        return(len(self.stack) == 0 and len(self.buffer) == 1)

    def calculate_cost(self):
        '''
        Calculate cost according to Figure 1 
        in "Arc-Hybrid Non-Projective Dependency Parsing 
        with a Static-Dynamic Oracle"
        '''

        s0 = self.stack[-1] if len(self.stack) > 0 else None
        s1 = self.stack[-2] if len(self.stack) > 1 else None
        b0 = self.buffer[0] if len(self.buffer) > 0 else None
        beta = self.buffer.contents[1:] if len(self.buffer) > 0 else None

        if not self.transition_admissible(LEFT_ARC):
            left_cost = 1
        else:
            left_cost = len(self.rdeps[s0])  + int(self.father[s0] != b0 and s0 in self.rdeps[self.father[s0]])

        if not self.transition_admissible(RIGHT_ARC):
            right_cost = 1
        else:
            right_cost = len(self.rdeps[s0])  + int(self.father[s0] != s1 and s0 in self.rdeps[self.father[s0]])
        
        if not self.transition_admissible(SHIFT):
            shift_cost = 1
            shift_case = 0
        elif len([item for item in beta if self.proj_order[item] < self.proj_order[b0] and item > b0]) > 0:
            shift_cost = 0
            shift_case = 1
        else:
            shift_cost = len([d for d in self.rdeps[b0] if d in self.stack]) + int(s0 != None and self.father[b0] in self.stack.contents[:-1] and b0 in self.rdeps[self.father[b0]])
            shift_case = 2
        
        if not self.transition_admissible(SWAP):
            swap_cost = 1
        elif self.proj_order[s0] > self.proj_order[b0]:
            swap_cost = 0
            left_cost = right_cost = shift_cost = 1
        else:
            swap_cost = 1
        
        costs = (shift_cost, left_cost, right_cost, swap_cost)
        return costs, shift_case

    def calculate_proj_order(self, node):
        # Calculate the projective order a.k.a inorder traversal order
        (self.left_nodes[node]).sort()
        for i in self.left_nodes[node]:
            self.calculate_proj_order(i)
        
        self.proj_order[node] = self.porj_order_cnt
        self.porj_order_cnt += 1
        
        (self.right_nodes[node]).sort()
        for i in self.right_nodes[node]:
            self.calculate_proj_order(i)


# This is for Eliyahu Kiperwasser's paper
# This doesn't work when there are non-projective trees
# def get_gold_actions(sentence):
#     # Initialize an arc-hybrid system 
#     words2id = list(range(1, len(sentence)+1))
#     config = Configuration(words2id)

#     fathers = {}
#     rdeps_num = {}

#     for i in range(len(sentence)+1):
#         rdeps_num[i] = 0

#     for word in sentence:
#         fathers[int(word['id'])] = int(word['father'])
#         rdeps_num[int(word['father'])] += 1
    
#     gold_actions = []

#     while not config.is_empty():
#         if config.transition_admissible(("left", None)) and fathers[config.stack[-1]] == config.buffer[0]:
#             rdeps_num[config.buffer[0]] -= 1
#             config.apply_transition(("left", None))
#             gold_actions.append(1)
#         elif config.transition_admissible(("right", None)) and fathers[config.stack[-1]] == config.stack[-2] and rdeps_num[config.stack[-1]] == 0:
#             rdeps_num[config.stack[-2]] -= 1
#             config.apply_transition(("right", None))
#             gold_actions.append(2)
#         elif config.transition_admissible('shift'):
#             config.apply_transition('shift')
#             gold_actions.append(0)
#         config.pretty_print()
#         print(gold_actions)
    
#     return gold_actions