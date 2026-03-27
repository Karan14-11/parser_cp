from __future__ import annotations
import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Any



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s", "--start_symbol", type=str, default="ROOT",
        help="Start symbol of the grammar (default: ROOT)"
    )
    parser.add_argument(
        "--progress", action="store_true", default=False,
        help="Display a progress bar (requires tqdm)"
    )
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level",
        action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level",
        action="store_const", const=logging.WARNING
    )
    parser.add_argument(
        "--chart", action="store_true", default=False,
        help="Print the Earley chart after parsing"
    )
    parser.add_argument(
        "--debug_p", action="store_true", default=False,
        help="Print debugging statements during parsing (for checking algorithm correctness)"
    )
    parser.add_argument(
        "--span", action="store_true", default=False,
        help="Print debugging statements during parsing (for checking algorithm correctness)"
    )
    return parser.parse_args()


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rule:
    """A grammar rule with a left-hand side, right-hand side, and weight (-log2 prob)."""
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        return f"{self.lhs} → {' '.join(self.rhs)}"


@dataclass(frozen=True)
class Item:
    """An Earley item: a dotted rule together with a start position.

    Frozen so it is hashable and can be used as a dictionary key.
    The item's identity (for duplicate detection) is determined by
    (rule, dot_position, start_position).
    """
    rule: Rule
    dot_position: int
    start_position: int

    def next_symbol(self) -> Optional[str]:
        """Return the symbol after the dot, or None if the dot is at the end."""
        if self.dot_position == len(self.rule.rhs):
            return None
        return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        """Return a new Item with the dot moved one position to the right."""
        if self.next_symbol() is None:
            raise IndexError("Can't advance dot past end of rule")
        return Item(
            rule=self.rule,
            dot_position=self.dot_position + 1,
            start_position=self.start_position
        )

    def __repr__(self) -> str:
        DOT = "·"
        rhs = list(self.rule.rhs)
        rhs.insert(self.dot_position, DOT)
        return f"({self.start_position}, {self.rule.lhs} → {' '.join(rhs)})"


# ─── Agenda ───────────────────────────────────────────────────────────────────

class Agenda:

    def __init__(self) -> None:
        self._items: List[Item] = []          # all items in push order
        self._index: Dict[Item, int] = {}     # item → index in _items (O(1) lookup)
        self._next: int = 0                   # next item to pop
        self._weights: Dict[Item, float] = {}
        self._backpointers: Dict[Item, Any] = {}

    def __len__(self) -> int:
        """Number of items waiting to be popped."""
        return len(self._items) - self._next

    def push(self, item: Item, weight: float, backpointer: Any = None) -> None:
        """Add item, or update weight/backpointer if a strictly better weight is found.
        O(1) amortized thanks to dict-based duplicate detection."""
        if item in self._index:
            # Item already seen – keep the better (lower) weight.
            if weight < self._weights[item]:
                self._weights[item] = weight
                self._backpointers[item] = backpointer
        else:
            self._items.append(item)
            self._index[item] = len(self._items) - 1
            self._weights[item] = weight
            self._backpointers[item] = backpointer

    def pop(self) -> Item:
        """Dequeue the next unprocessed item (FIFO order)."""
        if len(self) == 0:
            raise IndexError("Agenda is empty")
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """All items ever pushed (including already-popped ones).
        Needed for attach: completed items must find their customers."""
        return self._items

    def get_weight(self, item: Item) -> float:
        return self._weights[item]

    def get_backpointer(self, item: Item) -> Any:
        return self._backpointers.get(item)

    def __repr__(self) -> str:
        n = self._next
        return f"Agenda({self._items[:n]}; {self._items[n:]})"


# ─── Grammar ─────────────────────────────────────────────────────────────────

class Grammar:
    """A weighted context-free grammar loaded from .gr files."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Load rules from a tab-delimited .gr file.
        Format per line: <probability>\\t<lhs>\\t<rhs>
        Weight = -log2(probability)."""
        with open(file, "r") as f:
            for line in f:
                line = line.split("#")[0].rstrip()  # strip comments & trailing whitespace
                if line == "":
                    continue
                parts = line.split("\t")
                # print(parts)
                if len(parts) < 3:
                    continue
                prob_str, lhs, rhs_str = parts[0].strip(' '), parts[1].strip(' '), parts[2]
                prob = float(prob_str)
                rhs = tuple(rhs_str.split())
                if prob <= 0:
                    weight = float('inf')
                else:
                    weight = -math.log2(prob)
                rule = Rule(lhs=lhs, rhs=rhs, weight=weight)
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return all rules that expand `lhs`, or empty list if none."""
        return self._expansions.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        """A symbol is a nonterminal iff it has at least one expansion rule."""
        return symbol in self._expansions


# ─── Earley Chart Parser ─────────────────────────────────────────────────────

class EarleyChart:

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()
        self.cols: List[Agenda] = []
        self._run_earley()  # fill the chart

    def accepted(self) -> bool:
        """Was the sentence accepted (i.e. does a complete ROOT item exist)?"""
        return any(
            item.rule.lhs == self.grammar.start_symbol 
            and item.next_symbol() is None 
            and item.start_position == 0 
            for item in self.cols[-1].all()
        )

    def get_best_parse(self,args) -> Optional[Tuple[float, str]]:
        """Return (weight, tree_string) for the best (lowest-weight) parse, or None."""

        root_items = [
            (self.cols[-1].get_weight(it), it) for it in self.cols[-1].all()
            if it.rule.lhs == self.grammar.start_symbol and it.next_symbol() is None and it.start_position == 0
        ]
        # print(root_items)

        if not root_items:
            return None
        

        
        best_weight, best_item = min(root_items, key=lambda x: x[0])
        return best_weight, self._build_tree(best_item, len(self.tokens),args)
    
    def get_all_parse(self,args) -> List[Tuple[float, str]]:
        """Return a list of (weight, tree_string) for all successful root derivations."""
        
        # Identify all items that complete the start symbol across the whole span
        root_items = [
            (self.cols[-1].get_weight(it), it) 
            for it in self.cols[-1].all()
            if it.rule.lhs == self.grammar.start_symbol 
            and it.next_symbol() is None
            and it.start_position == 0
        ]

        if not root_items:
            return []

        # Map each root item to its reconstructed tree
        return [
            (weight, self._build_tree(item, len(self.tokens),args))
            for weight, item in root_items
        ]
    # ── Tree reconstruction via backpointers ──────────────────────────────



    def _build_tree(self, item: Item, col_idx: int,args) -> str:

        if args.span:

            """
            Reconstructs the tree string including spans: (LHS[start-end] children)
            """
            children: List[str] = []
            curr, idx = item, col_idx

            # Follow the backpointer chain to collect all children of this rule
            while curr.dot_position > 0:
                # get_backpointer returns (prev_item, prev_idx, child_info)
                prev_item, prev_idx, child_info = self.cols[idx].get_backpointer(curr)
                
                if isinstance(child_info, str):
                    # SCAN: child_info is the terminal token string
                    # A terminal at position i has span [i, i+1]
                    child = f"{child_info}[{idx-1}-{idx}]"
                else:
                    # ATTACH: child_info is (completed_item, completion_col_idx)
                    # This recursively builds the subtree for the completed nonterminal
                    child = self._build_tree(*child_info,args=args)
                
                children.append(child)
                curr, idx = prev_item, prev_idx

            # Join children in correct order and add the span [start-end] to the LHS
            content = ' '.join(reversed(children))
            return f"({item.rule.lhs}[{item.start_position}-{col_idx}] {content})"
        else:

            children: List[str] = []
            curr, idx = item, col_idx

            while curr.dot_position > 0:

                prev_item, prev_idx, child_info = self.cols[idx].get_backpointer(curr)
                
                # Use a ternary or type-check to handle SCAN vs ATTACH
                child = child_info if isinstance(child_info, str) else self._build_tree(*child_info,args=args)
                children.append(child)
                
                curr, idx = prev_item, prev_idx

            

            return f"({item.rule.lhs} {' '.join(reversed(children))})"
        

    def _enumerate_all_trees(self, item: Item, col_idx: int) -> List[Tuple[float, List[str], List[str]]]:
        """Enumerate all derivations for item ending at col_idx.
        Returns list of (weight, children_list, children_with_spans_list).
        Each entry represents one full derivation of all children of this item's rule."""
        if item.dot_position == 0:
            # Base case: no children consumed yet
            return [(0.0, [], [])]

        # Get ALL backpointers for this (item, col_idx) pair
        all_bp = self.cols[col_idx].get_all_backpointers(item)
        if not all_bp:
            bp = self.cols[col_idx].get_backpointer(item)
            if bp is None:
                return []
            all_bp = [(self.cols[col_idx].get_weight(item), bp)]

        results = []
        for bp_weight, bp in all_bp:
            prev_item, prev_col, child_info = bp

            if isinstance(child_info, str):
                # SCAN: the last child is a terminal
                child_str = child_info
                child_span_str = child_info
            else:
                # ATTACH: the last child is a completed nonterminal subtree
                attached_item, attached_col = child_info
                # Get all subtrees for the attached constituent
                sub_derivations = self._enumerate_all_trees_for_item(attached_item, attached_col)
                if not sub_derivations:
                    continue
                for (sub_w, sub_tree, sub_tree_spans) in sub_derivations:
                    # Recurse on prefix
                    prefix_derivations = self._enumerate_all_trees(prev_item, prev_col)
                    for (pw, prefix_children, prefix_span_children) in prefix_derivations:
                        results.append((
                            bp_weight,
                            prefix_children + [sub_tree],
                            prefix_span_children + [sub_tree_spans]
                        ))
                continue  # already added results for ATTACH case

            # For SCAN case: recurse on prefix
            prefix_derivations = self._enumerate_all_trees(prev_item, prev_col)
            for (pw, prefix_children, prefix_span_children) in prefix_derivations:
                results.append((
                    bp_weight,
                    prefix_children + [child_str],
                    prefix_span_children + [child_span_str]
                ))

        return results

    # ── Core Earley algorithm ────────────────────────────────────────────

    def _run_earley(self) -> None:
        """Fill in the Earley chart using Predict/Scan/Attach."""
        n = len(self.tokens)
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]
        self._predict(self.grammar.start_symbol, 0)

        # Process columns left to right

        for i, column in enumerate(self.cols):
            while column:
                item = column.pop()
                next_sym = item.next_symbol()
                
                # Use a dispatch-like logic flow
                if next_sym is None:
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next_sym):
                    self._predict(next_sym, i)
                else:
                    self._scan(item, i)


    def _predict(self, nonterminal: str, position: int) -> None:

        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            self.cols[position].push(new_item, weight=rule.weight, backpointer=None)
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:

        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            self.cols[position + 1].push(
                item.with_dot_advanced(),
                weight=self.cols[position].get_weight(item),
                backpointer=(item, position, self.tokens[position])
            )
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:

        completed_weight = self.cols[position].get_weight(item)

        for customer in self.cols[item.start_position].all():
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                customer_weight = self.cols[item.start_position].get_weight(customer)
                new_weight = customer_weight + completed_weight
                bp = (customer, item.start_position, (item, position))  # backpointer: attached constituent
                self.cols[position].push(new_item, weight=new_weight, backpointer=bp)
                self.profile["ATTACH"] += 1

    # ── Chart printing ───────────────────────────────────────────────────

    def print_chart(self) -> None:
        """Print the Earley chart (all columns and their items)."""
        for i, col in enumerate(self.cols):
            if i == 0:
                print(f"\n--- Column {i} (before input) ---")
            elif i <= len(self.tokens):
                print(f"\n--- Column {i} (after '{self.tokens[i-1]}') ---")
            else:
                print(f"\n--- Column {i} ---")
            for item in col.all():
                w = col.get_weight(item)
                print(f"  {item}  [w={w:.4f}]")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if args.debug_p:
                print(sentence)
            if sentence != "":   # skip blank lines
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                if args.chart:
                    chart.print_chart()
                result = chart.get_best_parse(args=args)
                if result is not None:
                    weight, tree = result
                    print(tree)
                    print(f"{weight}")
                    print(f"Probability: {2 ** -weight}")

                # result = chart.get_all_parse(args=args)
                # if result is not None:
                #     tree = result
                #     print(tree)
                #     print(f"{weight}")
                #     # print(f"Probability: {2 ** -weight}")
                # else:
                #     print("NONE")

                


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    main()