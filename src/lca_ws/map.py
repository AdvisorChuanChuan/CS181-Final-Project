import util

class Mymap:
    def __init__(self):
        self.destination_idx = 0
        self.restaurants_num = 4
        self.des_pos = (5,2)
        self.restaurants_poss = [(0,0), (1,8), (6,6), (13,4)]
        self.accessed = []
        self.positions = []
        self.successors = {}
        self.init_positions()
        self.init_access()
        self.init_successor()

    def init_positions(self):
        for i in range(0, 9):
            self.positions.append((0, i))
        for i in range(0, 9):
            self.positions.append((1, i))
        for i in range(0, 9):
            self.positions.append((2, i))
        self.positions.append((3, 2))
        self.positions.append((3, 4))
        self.positions.append((3, 8))
        for i in range(0, 9):
            self.positions.append((4, i))
        for i in range(2, 9):
            self.positions.append((5, i))
        self.positions.append((5, 0))
        for i in range(4, 9):
            self.positions.append((6, i))
        self.positions.append((6, 0))
        for i in range(0, 9):
            self.positions.append((7, i))
        for i in range(8, 14):
            for j in range(4, 9):
                self.positions.append((i, j))

    def init_access(self):
        # row
        for i in range(0, 8):
            self.accessed.append(((0, i), (0, i + 1)))
        for i in range(0, 8):
            self.accessed.append(((1, i), (1, i + 1)))
        for i in range(0, 8):
            self.accessed.append(((2, i), (2, i + 1)))
        for i in range(0, 8):
            self.accessed.append(((4, i), (4, i + 1)))
        for i in range(2, 8):
            self.accessed.append(((5, i), (5, i + 1)))
        for i in range(5, 8):
            self.accessed.append(((6, i), (6, i + 1)))
        for i in range(0, 8):
            self.accessed.append(((7, i), (7, i + 1)))
        for i in range(4, 8):
            self.accessed.append(((9, i), (9, i + 1)))
        for i in range(4, 8):
            self.accessed.append(((11, i), (11, i + 1)))
        for i in range(4, 8):
            self.accessed.append(((13, i), (13, i + 1)))
        # col
        for i in range(0, 2):
            self.accessed.append(((i, 0), (i + 1, 0)))
        for i in range(4, 7):
            self.accessed.append(((i, 0), (i + 1, 0)))
        for i in range(0, 4):
            self.accessed.append(((i, 2), (i + 1, 2)))
        for i in range(0, 13):
            self.accessed.append(((i, 4), (i + 1, 4)))
        for i in range(4, 13):
            self.accessed.append(((i, 5), (i + 1, 5)))
        for i in range(4, 13):
            self.accessed.append(((i, 6), (i + 1, 6)))
        for i in range(4, 13):
            self.accessed.append(((i, 7), (i + 1, 7)))
        for i in range(0, 13):
            self.accessed.append(((i, 8), (i + 1, 8)))

    def init_successor(self):
        for pos in self.positions:
            self.successors[pos] = []
            for access in self.accessed:
                x, y = access
                if x == pos:
                    pos_next = y
                elif y == pos:
                    pos_next = x
                else:
                    continue
                successor = self.successors[pos]
                successor.append(pos_next)
                self.successors[pos] = successor

    def get_successor(self, x):
        return self.successors[x]

    def bfs(self, start, destination):
        past = []
        output_list = []
        queue = util.Queue()
        queue.push((start, []))
        while not queue.isEmpty():
            node, output_list = queue.pop()
            if node == destination:
                break
            if node not in past:
                past.append(node)
                leaf_successors = self.get_successor(node)
                for successor in leaf_successors:
                    if successor not in past:
                        queue.push((successor, output_list + [successor]))
        return output_list


# world_map = Mymap()
# print(len(world_map.bfs((5, 0), (9, 4))))