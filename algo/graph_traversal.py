from collections import defaultdict


def move(curr, other, visited_neighbours, back):
  visited_neighbours[curr].add(other)
  # only set the backward arc the first time we enter vertex
  if other not in back:
    back[other] = curr
  return other


def traverse_graph(g):
  # assume every arc xy has a backward arc yx
  # assume the graph is connected
  # the algorithm below traverses every arc exactly once
  # as a consequence, it traverses every vertex
  curr = 0
  back = {0 : None}
  visited_neighbours = defaultdict(set)
  while True:
    moved = False
    # try to traverse any untraversed arc, besides the backward one
    for other in g[curr]:
      if other != back[curr] and other not in visited_neighbours[curr]:
        curr = move(curr, other, visited_neighbours, back)
        moved = True
        break
    # backward arc should be the last one to traverse
    if not moved and back[curr] is not None:
      curr = move(curr, back[curr], visited_neighbours, back)
      moved = True
    if not moved:
      break
  return visited_neighbours


def test_traverse_graph():
  g = [[1],
       [2, 3, 4, 0],
       [1, 3],
       [1, 2],
       [1, 5],
       [4]]
  visited_neighbours = traverse_graph(g)
  for index in xrange(len(g)):
    assert index in visited_neighbours


if __name__ == '__main__':
  test_traverse_graph()
