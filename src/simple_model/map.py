import util

class Mymap:
    def __init__(self):
        self.destination_idx = 0
        self.restaurants_num = 4
        self.des_pos = 0
        self.restaurants_poss = [10]
        self.accessed = []
        self.positions = []
        self.successors = {}
        self.init_positions()
        self.init_access()
        self.init_successor()

    def init_positions(self):
        for i in range(11):
            self.positions.append(i)

    def init_access(self):
        for i in range(10):
            self.accessed.append((i, i + 1))

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

    def search_route(self, restaurants, pos):
        path = []
        path_length = []
        if len(restaurants) == 0:
            des_path = len(self.bfs(pos, self.des_pos))
            return [None, des_path]
        for restaurant in restaurants:
            path.append(self.bfs(pos, restaurant))
            new_res = []
            for restaurant_copy in restaurants:
                new_res.append(restaurant_copy)
            new_res.remove(restaurant)
            path_length.append(len(self.bfs(pos, restaurant)) + self.search_route(new_res, restaurant)[1])
        path_min = min(path_length)
        path_next_res = path_length.index(path_min)
        return [path_next_res, path_min]


world_map = Mymap()
print(world_map.bfs(0, 9))