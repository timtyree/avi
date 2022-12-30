import numpy as np
# Timothy Tyree
# 12.4.2020

#Ready-Made Example

from queue import PriorityQueue

q = PriorityQueue()

q.put((4, 'Read'))
q.put((2, 'Play'))
q.put((5, 'Write'))
q.put((1, 'Code'))
q.put((3, 'Study'))

while not q.empty():
    next_item = q.get()
    print(next_item)

# the following parallel priority queue is very fast, but it makes use of bucket heaps, which I haven't found a ready-made soln for...
# "A parallel priority queue with fast updates for GPU architectures" by John Iacono et al. (2019)
# `parallel priority queue.pdf` <<prereq?: `bucket heap paper.pdf`, which looks complicated>>
