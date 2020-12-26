class Order:
    def __init__(self, _order_idx, _startTime, _endTime, _price, _restaurant_idx):
        self.index = _order_idx
        self.startTime = _startTime
        self.endTime = _endTime
        self.price = _price
        self.restaurant_idx = _restaurant_idx
