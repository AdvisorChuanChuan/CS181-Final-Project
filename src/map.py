class Mymap:
    def __init__(self):
        self.destination_idx = 0
        self.restaurants_num = 4
        self.restaurants_idxs = [i for i in range(1, self.restaurants_num+1)]
        self.walls = []  # TODO