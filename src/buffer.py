class OrderBuffer:
    def __init__(self):
        self.untreated_orders = []
        self.accepted_orders = []

    def pushNewOrder(self, _order):
        self.untreated_orders.append(_order)
    
    def acceptOrder(self, _order_idx):
        