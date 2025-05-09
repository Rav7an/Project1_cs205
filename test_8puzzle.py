import time
import matplotlib.pyplot as plt
from n_Puzzle import general_search, ucs_queue, astar_misplaced_queue, astar_manhattan_queue

boards = [
    [1,2,3,4,5,6,7,8,0],
    [1,2,3,4,5,6,0,7,8],
    [1,2,3,5,0,6,4,7,8],
    [1,3,6,5,0,2,4,7,8],
    [1,3,6,5,0,7,4,8,2],
    [1,6,7,5,0,3,4,8,2],
    [7,1,2,4,8,5,6,3,0],
    [0,7,2,4,6,1,3,5,8],
]

algs = [
    ("UCS", ucs_queue),
    ("A* Misplaced", astar_misplaced_queue),
    ("A* Manhattan", astar_manhattan_queue),
]

results = {name: {'d': [], 'n': [], 't': [], 'q': []} for name,_ in algs}

for board in boards:
    for name, fn in algs:
        metrics = {'n': 0, 'q': 0}
        def instrument(open_list, children):
            metrics['n'] += 1
            metrics['q'] = max(metrics['q'], len(open_list))
            fn(open_list, children)

        t0 = time.perf_counter()
        sol = general_search(board, instrument)
        t1 = time.perf_counter()

        depth = sol.depth if sol else None
        results[name]['d'].append(depth)
        results[name]['n'].append(metrics['n'])
        results[name]['t'].append(t1 - t0)
        results[name]['q'].append(metrics['q'])

for metric, label in [('n','Nodes Expanded'), ('t','Time (s)'), ('q','Max Queue')] :
    plt.figure()
    for name in results:
        plt.plot(results[name]['d'], results[name][metric], marker='o', label=name)
    plt.xlabel('Solution Depth')
    plt.ylabel(label)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
plt.show()
