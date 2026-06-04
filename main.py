import argparse
import csv
import sys
from typing import List, Tuple

from cfpq_matrix import Grammar, LabeledGraph, cfpq_matrix
from cfpq_kronecker import cfpq_kronecker
from grammar_utils import parse_and_normalize
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

#граф с двумя циклами, грамматика a^n b^n

def example_paper() -> None:
    print("=" * 60)
    print("двойной цикл, L = {a^n b^n | n >= 1}")
    print("=" * 60)

    edges: List[Tuple[int, str, int]] = [
        (0, "a", 1),
        (1, "a", 2),
        (2, "a", 0),
        (0, "b", 3),
        (3, "b", 0),
    ]
    graph = LabeledGraph.from_edges(edges)

    grammar_cnf_text = """
S -> A B
S -> A S1
S1 -> S B
A -> a
B -> b
"""
    grammar = Grammar.from_text(grammar_cnf_text)

    print("\n[Алгоритм 1: матричное умножение]")
    results = cfpq_matrix(graph, grammar, use_sparse=True)
    print(f"  R_S  = {sorted(results['S'])}")
    print(f"  R_S1 = {sorted(results['S1'])}")
    print(f"  R_A  = {sorted(results['A'])}")
    print(f"  R_B  = {sorted(results['B'])}")

    # Ожидаемый результат из статьи:
    # R_S = {(0,0),(0,3),(1,0),(1,3),(2,0),(2,3)}
    expected_S = {(0, 0), (0, 3), (1, 0), (1, 3), (2, 0), (2, 3)}
    got_S = set(results["S"])
    status = "+ВЕРНО+" if got_S == expected_S else f"Ожидалось {expected_S}"
    print(f"\n  Проверка R_S: {status}")

    print("\n[Алгоритм Кронекера]")
    grammar_text = """
S -> a S b
S -> a b
"""
    kron_results = cfpq_kronecker(edges, graph.num_nodes, grammar_text, "S")
    print(f"  R_S (Кронекер) = {sorted(kron_results)}")


def example_object_model() -> None:
    """
    Пример анализа объектной модели (граф зависимостей классов).
    Вершины: 0=BaseClass, 1=MidClass, 2=TopClass, 3=Component
    Метки рёбер: subclassOf, fieldOf
    Грамматика: находим все пары, связанные транзитивным наследованием
    """
    print("\n" + "=" * 60)
    print("Пример: объектная модель, транзитивное наследование")
    print("=" * 60)

    edges: List[Tuple[int, str, int]] = [
        (0, "subclassOf", 1),
        (1, "subclassOf", 2),
        (0, "fieldOf", 3),
        (3, "fieldOf", 1),
    ]
    graph = LabeledGraph.from_edges(edges)
    node_names = {0: "Base", 1: "Mid", 2: "Top", 3: "Component"}

    grammar_text = """
Inherit -> subclassOf Inherit
Inherit -> subclassOf
"""
    grammar = parse_and_normalize(grammar_text, start="Inherit")

    print(f"\nГраф: {edges}")
    print(f"Правила: транзитивное subclassOf")

    results = cfpq_matrix(graph, grammar, start_nonterminal="Inherit", use_sparse=False)
    pairs = results.get("Inherit", [])
    print("\nПути по наследованию:")
    for i, j in sorted(pairs):
        print(f"  {node_names[i]} -->* {node_names[j]}")



def example_same_generation() -> None:
  
    print("\n" + "=" * 60)
    print("Пример: same-generation query")
    print("=" * 60)

    edges: List[Tuple[int, str, int]] = [
        (0, "subClassOf", 1),
        (1, "subClassOf", 2),
        (2, "subClassOf", 3),
        (3, "subClassOf", 0),   # цикл
        (0, "subClassOf_r", 1),  # обратные рёбра (_r = inverse)
        (1, "subClassOf_r", 0),
        (2, "subClassOf_r", 1),
        (3, "subClassOf_r", 2),
        (0, "subClassOf_r", 3),
    ]
    graph = LabeledGraph.from_edges(edges)

    grammar_cnf = """
S -> SC_R_NT S1
S1 -> S SC_NT
S -> SC_R_NT SC_NT
SC_NT -> subClassOf
SC_R_NT -> subClassOf_r
"""
    grammar = Grammar.from_text(grammar_cnf)

    print(f"\nВершин: {graph.num_nodes}, рёбер: {len(graph.edges)}")
    results = cfpq_matrix(graph, grammar, use_sparse=True)
    print(f"R_S (same-generation) = {sorted(results.get('S', []))}")


def load_graph_from_csv(path: str) -> Tuple[List[Tuple[int, str, int]], int]:
    edges = []
    max_node = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            src, label, dst = int(row[0]), row[1].strip(), int(row[2])
            edges.append((src, label, dst))
            max_node = max(max_node, src, dst)
    return edges, max_node + 1


def load_grammar_from_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="CFPQ — контекстно-свободный поиск путей в графе"
    )
    parser.add_argument(
        "--example",
        choices=["paper", "model", "samegeneration", "all"],
        default="all",
        help="Запустить встроенный пример",
    )
    parser.add_argument("--graph", help="Путь к CSV-файлу графа (src,label,dst)")
    parser.add_argument("--grammar", help="Путь к файлу грамматики")
    parser.add_argument("--start", default="S", help="Стартовый нетерминал (по умолчанию S)")
    parser.add_argument(
        "--algo",
        choices=["matrix", "kronecker", "both"],
        default="matrix",
        help="Алгоритм: matrix (по умолчанию) или kronecker",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=True,
        help="Использовать разреженные матрицы (по умолчанию True)",
    )

    args = parser.parse_args()

    if args.graph and args.grammar:
        # Пользовательский граф
        edges, n = load_graph_from_csv(args.graph)
        grammar_text = load_grammar_from_file(args.grammar)
        print(f"Граф: {n} вершин, {len(edges)} рёбер")
        print(f"Стартовый нетерминал: {args.start}")

        if args.algo in ("matrix", "both"):
            grammar = parse_and_normalize(grammar_text, start=args.start)
            graph = LabeledGraph(num_nodes=n, edges=edges)
            results = cfpq_matrix(graph, grammar, start_nonterminal=args.start, use_sparse=args.sparse)
            print(f"\n[Matrix] R_{args.start} = {sorted(results.get(args.start, []))}")

        if args.algo in ("kronecker", "both"):
            pairs = cfpq_kronecker(edges, n, grammar_text, start_nonterminal=args.start)
            print(f"\n[Kronecker] R_{args.start} = {sorted(pairs)}")

    else:
        # Встроенные примеры
        if args.example in ("paper", "all"):
            example_paper()
        if args.example in ("model", "all"):
            example_object_model()
        if args.example in ("samegeneration", "all"):
            example_same_generation()


if __name__ == "__main__":
    main()