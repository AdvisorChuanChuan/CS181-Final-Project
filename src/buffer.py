class OrderBuffer:
    def __init__(self):
        self.untreated_orders = []
        self.recieved_orders = []

    def pushNewOrder(self, order):
        self.untreated_orders.append(order)