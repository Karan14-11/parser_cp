#!/usr/bin/env python3
"""
A refactored Probabilistic Earley Parser. 
Corrected to support both Viterbi (best-path) and full Forest (all-parses) reconstruction.
"""

from __future__ import annotations
import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Any

# --- Configuration & Logging ---
logger = logging.getLogger("EarleyParser")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar", type=Path, help="PCFG .gr file")
    parser.add_argument("sentences", type=Path, help="Tokenized .sen file")
    parser.add_argument("-s", "--start", type=str, default="ROOT", help="Start symbol")
    parser.add_argument("--chart", action="store_true", help="Display Earley chart")
    parser.add_argument("--all-parses", action="store_true", help="Print all valid derivations")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, dest="log_level")
    group.add_argument("-q", "--quiet", action="store_const", const=logging.WARNING, dest="log_level")
    parser.set_defaults(log_level=logging.INFO)
    
    return parser.parse_args()

@dataclass(frozen=True)
class Production:
    head: str
    body: Tuple[str, ...]
    weight: float = 0.0

@dataclass(frozen=True)
class State:
    rule: Production
    dot: int
    origin: int

    @property
    def next_sym(self) -> Optional[str]:
        return self.rule.body[self.dot] if self.dot < len(self.rule.body) else None

    @property
    def is_complete(self) -> bool:
        return self.dot == len(self.rule.body)

    def advance(self) -> State:
        return State(self.rule, self.dot + 1, self.origin)

class ChartColumn:
    def __init__(self):
        self.states: List[State] = []
        self._state_map: Dict[State, int] = {}
        self._cursor: int = 0
        self.best_weights: Dict[State, float] = {}
        self.viterbi_bp: Dict[State, Any] = {}
        # Stores ALL incoming paths: (weight_of_step, backpointer_data)
        self.all_paths: Dict[State, List[Tuple[float, Any]]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self.states) - self._cursor

    def add(self, state: State, weight: float, bp: Any = None) -> None:
        # Viterbi logic: Keep track of the absolute best weight for this state
        if state not in self._state_map:
            self._state_map[state] = len(self.states)
            self.states.append(state)
            self.best_weights[state] = weight
            self.viterbi_bp[state] = bp
        elif weight < self.best_weights[state]:
            self.best_weights[state] = weight
            self.viterbi_bp[state] = bp

        # Forest logic: Always record the path if a backpointer exists
        if bp is not None:
            self.all_paths[state].append((weight, bp))

    def consume(self) -> State:
        state = self.states[self._cursor]
        self._cursor += 1
        return state

class PCFG:
    def __init__(self, start_node: str, file_path: Path):
        self.start_node = start_node
        self.rules: Dict[str, List[Production]] = defaultdict(list)
        self._load(file_path)

    def _load(self, path: Path):
        with open(path, "r") as f:
            for line in f:
                clean = line.split("#")[0].strip()
                if not clean: continue
                parts = clean.split("\t")
                if len(parts) < 3: continue
                p_val, lhs, rhs_val = float(parts[0]), parts[1].strip(), tuple(parts[2].split())
                w = -math.log2(p_val) if p_val > 0 else float('inf')
                self.rules[lhs].append(Production(lhs, rhs_val, w))

    def is_nt(self, symbol: str) -> bool:
        return symbol in self.rules

class Parser:
    def __init__(self, grammar: PCFG, tokens: List[str]):
        self.grammar = grammar
        self.tokens = tokens
        self.chart = [ChartColumn() for _ in range(len(tokens) + 1)]
        self._execute()

    def _execute(self):
        for rule in self.grammar.rules[self.grammar.start_node]:
            self.chart[0].add(State(rule, 0, 0), rule.weight)

        for i in range(len(self.tokens) + 1):
            col = self.chart[i]
            while col:
                st = col.consume()
                if st.is_complete:
                    self._attach(st, i)
                elif self.grammar.is_nt(st.next_sym):
                    self._predict(st.next_sym, i)
                else:
                    self._scan(st, i)

    def _predict(self, nt: str, pos: int):
        for rule in self.grammar.rules[nt]:
            self.chart[pos].add(State(rule, 0, pos), rule.weight)

    def _scan(self, st: State, pos: int):
        if pos < len(self.tokens) and st.next_sym == self.tokens[pos]:
            # Weight is carried over from the state before scanning the terminal
            w = self.chart[pos].best_weights[st]
            self.chart[pos + 1].add(st.advance(), w, (st, pos, self.tokens[pos]))

    def _attach(self, st: State, pos: int):
        mid = st.origin
        # For every customer waiting for this completed non-terminal
        for customer in self.chart[mid].states:
            if customer.next_sym == st.rule.head:
                # We must consider ALL paths that could have formed this completion
                # For Viterbi, we use best_weights; for all_parses, we'll recurse later.
                for path_w, _ in self.chart[pos].all_paths[st]:
                    total_w = self.chart[mid].best_weights[customer] + path_w
                    self.chart[pos].add(customer.advance(), total_w, (customer, mid, (st, pos)))

    def get_results(self, all_parses: bool) -> List[Tuple[float, str, str]]:
        final_col = self.chart[len(self.tokens)]
        seeds = [s for s in final_col.states 
                 if s.rule.head == self.grammar.start_node and s.is_complete and s.origin == 0]
        if not seeds: return []

        if not all_parses:
            best = min(seeds, key=lambda x: final_col.best_weights[x])
            return [self._extract_viterbi(best, len(self.tokens))]
        
        all_found = []
        for s in seeds:
            all_found.extend(self._extract_all(s, len(self.tokens)))
        return sorted(all_found, key=lambda x: x[0])

    def _extract_viterbi(self, state: State, col_idx: int) -> Tuple[float, str, str]:
        def build(curr_st, curr_col, spans):
            if curr_st.dot == 0: return []
            bp = self.chart[curr_col].viterbi_bp[curr_st]
            prev_st, prev_col, data = bp
            if isinstance(data, str):
                child = data
            else:
                child_st, child_col = data
                child = build_tree(child_st, child_col, spans)
            return build(prev_st, prev_col, spans) + [child]

        def build_tree(s, c, spans):
            kids = build(s, c, spans)
            tag = f"{s.rule.head} [{s.origin},{c}]" if spans else s.rule.head
            return f"({tag} {' '.join(kids)})"

        w = self.chart[col_idx].best_weights[state]
        return (w, build_tree(state, col_idx, False), build_tree(state, col_idx, True))

    def _extract_all(self, state: State, col_idx: int) -> List[Tuple[float, str, str]]:
        memo_derivs = {}
        memo_subtrees = {}

        def get_derivs(curr_st, curr_col):
            state_key = (curr_st, curr_col)
            if curr_st.dot == 0: return [(curr_st.rule.weight, [], [])]
            if state_key in memo_derivs: return memo_derivs[state_key]
            
            res = []
            for _, bp in self.chart[curr_col].all_paths[curr_st]:
                prev_st, prev_col, data = bp
                if isinstance(data, str): # Scan
                    for pw, pc, ps in get_derivs(prev_st, prev_col):
                        res.append((pw, pc + [data], ps + [data]))
                else: # Attach
                    c_st, c_col = data
                    for cw, ct, cs in get_subtrees(c_st, c_col):
                        for pw, pc, ps in get_derivs(prev_st, prev_col):
                            res.append((pw + cw, pc + [ct], ps + [cs]))
            memo_derivs[state_key] = res
            return res

        def get_subtrees(s, c):
            state_key = (s, c)
            if state_key in memo_subtrees: return memo_subtrees[state_key]
            derivs = get_derivs(s, c)
            unique = {}
            for w, kids, skids in derivs:
                t = f"({s.rule.head} {' '.join(kids)})"
                ts = f"({s.rule.head} [{s.origin},{c}] {' '.join(skids)})"
                if t not in unique or w < unique[t][0]:
                    unique[t] = (w, t, ts)
            res = list(unique.values())
            memo_subtrees[state_key] = res
            return res

        return get_subtrees(state, col_idx)

def main():
    args = get_args()
    grammar = PCFG(args.start, args.grammar)
    with open(args.sentences, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens: continue
            parser = Parser(grammar, tokens)
            results = parser.get_results(args.all_parses)
            if not results:
                print("NONE")
            else:
                for weight, tree, span_tree in results:
                    print(f"{tree}\n{span_tree}\n{weight:.6f}\n{2**-weight:.8f}")

if __name__ == "__main__":
    main()