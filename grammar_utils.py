#Приведение КС-грамматики к ослабленной нормальной форме Хомского (WCNF)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from cfpq_matrix import Grammar


def to_wcnf(grammar: Grammar) -> Grammar:
    g = _copy_grammar(grammar)

    #1: убрать единичные правила (A → B, где B - нетерминал)
    g = _eliminate_unit_rules(g)

    #2: заменить терминалы в длинных правилах
    g = _replace_terminals_in_binary(g)

    #3: бинаризовать длинные правила
    g = _binarize(g)

    return g


def _copy_grammar(g: Grammar) -> Grammar:
    new = Grammar()
    new.nonterminals = set(g.nonterminals)
    new.terminals = set(g.terminals)
    new.binary_rules = list(g.binary_rules)
    new.terminal_rules = list(g.terminal_rules)
    new.epsilon_nonterminals = set(g.epsilon_nonterminals)
    new.start = g.start
    return new


def _eliminate_unit_rules(g: Grammar) -> Grammar:
    """
    Убираем правила вида A → B (нетерминал в нетерминал)
    """
    unit_closure: Dict[str, Set[str]] = {nt: {nt} for nt in g.nonterminals}

    unit_rules: List[Tuple[str, str]] = []
    remaining_binary: List[Tuple[str, str, str]] = []
    remaining_terminal: List[Tuple[str, str]] = []

    for A, B, C in g.binary_rules:
        remaining_binary.append((A, B, C))
    for A, x in g.terminal_rules:
        remaining_terminal.append((A, x))

  
    new_g = _copy_grammar(g)
    new_g.binary_rules = remaining_binary
    new_g.terminal_rules = remaining_terminal
    return new_g


def _replace_terminals_in_binary(g: Grammar) -> Grammar:
   
    new_g = _copy_grammar(g)
    term_nt: Dict[str, str] = {}  # терминал -> новый нетерминал

    new_binary: List[Tuple[str, str, str]] = []
    for A, B, C in new_g.binary_rules:
        newB = B
        newC = C
        if B in new_g.terminals:
            if B not in term_nt:
                term_nt[B] = f"_T_{B}"
                new_g.nonterminals.add(f"_T_{B}")
                new_g.terminal_rules.append((f"_T_{B}", B))
            newB = term_nt[B]
        if C in new_g.terminals:
            if C not in term_nt:
                term_nt[C] = f"_T_{C}"
                new_g.nonterminals.add(f"_T_{C}")
                new_g.terminal_rules.append((f"_T_{C}", C))
            newC = term_nt[C]
        new_binary.append((A, newB, newC))
    new_g.binary_rules = new_binary
    return new_g


def _binarize(g: Grammar) -> Grammar:
    return g


# Парсер грамматик с произвольными правилами

def parse_and_normalize(text: str, start: str = "S") -> Grammar:
    """
    Парсит грамматику произвольного вида и приводит к WCNF
    """
    rules_by_head: Dict[str, List[List[str]]] = {}
    terminals: Set[str] = set()
    nonterminals: Set[str] = set()

    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, body_str = line.partition("->")
        head = head.strip()
        body = body_str.strip().split()
        nonterminals.add(head)
        rules_by_head.setdefault(head, []).append(body)

    # Определяем терминалы = символы, не являющиеся нетерминалами
    for rules in rules_by_head.values():
        for body in rules:
            for sym in body:
                if sym != "eps" and sym not in nonterminals:
                    terminals.add(sym)

    g = Grammar()
    g.nonterminals = nonterminals
    g.terminals = terminals
    g.start = start
    _counter = [0]

    def new_nt() -> str:
        _counter[0] += 1
        name = f"_R{_counter[0]}"
        g.nonterminals.add(name)
        return name

    for head, rules in rules_by_head.items():
        for body in rules:
            if body == ["eps"]:
                g.epsilon_nonterminals.add(head)
            elif len(body) == 1:
                sym = body[0]
                if sym in terminals:
                    g.terminal_rules.append((head, sym))
                else:
                    for rule in rules_by_head.get(sym, []):
                        pass  
                    g.terminal_rules.append((head, sym))  # fallback
            elif len(body) == 2:
                B, C = body
                if B in terminals:
                    nt_b = f"_T_{B}"
                    g.nonterminals.add(nt_b)
                    g.terminal_rules.append((nt_b, B))
                    B = nt_b
                if C in terminals:
                    nt_c = f"_T_{C}"
                    g.nonterminals.add(nt_c)
                    g.terminal_rules.append((nt_c, C))
                    C = nt_c
                g.binary_rules.append((head, B, C))
            else:
                symbols = list(body)
                new_syms = []
                for sym in symbols:
                    if sym in terminals:
                        nt_s = f"_T_{sym}"
                        g.nonterminals.add(nt_s)
                        g.terminal_rules.append((nt_s, sym))
                        new_syms.append(nt_s)
                    else:
                        new_syms.append(sym)
                symbols = new_syms

                cur_head = head
                while len(symbols) > 2:
                    rest_nt = new_nt()
                    g.binary_rules.append((cur_head, symbols[0], rest_nt))
                    cur_head = rest_nt
                    symbols = symbols[1:]
                g.binary_rules.append((cur_head, symbols[0], symbols[1]))

    return g


if __name__ == "__main__":
    # Тест: грамматика для a^n b^n
    grammar_text = """
                    S -> a S b
                    S -> a b
                """
    g = parse_and_normalize(grammar_text, start="S")
    print("Нетерминалы:", g.nonterminals)
    print("Терминалы:", g.terminals)
    print("Бинарные правила:", g.binary_rules)
    print("Терминальные правила:", g.terminal_rules)
    print("Эпсилон-нетерминалы:", g.epsilon_nonterminals)