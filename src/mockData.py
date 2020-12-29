import numpy as np
import pandas as pd
import datetime
import random
from src.constants import destinations, restaurants, foods, week


class MyTime(datetime.time):
    def __add__(self, other: datetime.timedelta):
        tem = datetime.datetime(2020, 1, 1, self.hour, self.minute)
        tem += other
        return MyTime(tem.hour, tem.minute)

    def __sub__(self, other):
        if type(other) == MyTime:
            tem_1 = datetime.datetime(2020, 1, 1, self.hour, self.minute)
            tem_2 = datetime.datetime(2020, 1, 1, other.hour, other.minute)
            return tem_1 - tem_2
        else:
            tem = datetime.datetime(2020, 1, 1, self.hour, self.minute)
            tem -= other
            return MyTime(tem.hour, tem.minute)


class Order:
    def __init__(self, created_time: datetime.time, restaurant: str, destination: str, food: str):
        self.create_time = created_time
        self.restaurant = restaurant
        self.destination = destination
        self.food = food


class OrderGenerator:
    # generate orders in one week
    def __init__(self, res, des, foo, size):
        self.restaurants = res
        self.destination = des
        self.foods = foo
        self.size = size
        self.dataSource = []
        self.weekInfo = dict(zip(range(1, 8), week))

        self.generate()

    def generate(self):
        for day in week:
            pass


if __name__ == "__main__":
    a = MyTime(11, 5)
    b = datetime.timedelta(minutes=5)
    c = a + b
    d = c - a
    instance = OrderGenerator(restaurants, destinations, foods, size=1e4)
