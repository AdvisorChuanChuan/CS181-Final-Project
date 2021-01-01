import numpy as np
import json
import os
import pandas as pd
import datetime
import sys
sys.path.append("..")
from constants import destinations, restaurants, foods, week


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
    def __init__(self, created_time: datetime.time, restaurant: str, destination: str, food: str = None):
        self.create_time = created_time
        self.restaurant = restaurant
        self.destination = destination
        self.food = food


class OrderGenerator:
    # generate orders in one week
    def __init__(self, res, des, foo, day):
        self.restaurants = res
        self.destination = des
        self.foods = foo
        self.size = day
        self.dataSource = []
        self.weekInfo = dict(zip(range(1, 8), week))

        for i in range(0, int(day)):
            self.generate_single_day(10)

        # self.csv_encoder()

    def csv_encoder(self):
        print("Simulate Days Number: {}".format(len(self.dataSource)))
        file_index = 0
        tem_create_time = []
        tem_restaurant = []
        if not os.path.exists("data"):
            os.makedirs("data")
        for day in self.dataSource:
            for data in day:
                tem_create_time.append(str(data.create_time))
                tem_restaurant.append(data.restaurant)
            data = pd.DataFrame(data={
                "created_time": tem_create_time,
                "restaurant": tem_restaurant
            })
            data.to_csv('./data/{}.csv'.format(file_index), index=True)
            file_index += 1

    def generate_single_day(self, size):
        tem_re = []
        features = [
            {
                "time": MyTime(11, 50),
                "res": self.restaurants[0]
            },
            {
                "time": MyTime(12, 30),
                "res": self.restaurants[1]
            },
            {
                "time": MyTime(11, 30),
                "res": self.restaurants[2]
            },
            {
                "time": MyTime(12, 50),
                "res": self.restaurants[3]
            }
        ]
        for f in features:
            for i in range(size):
                delta = np.random.normal(0, 5)
                tem_re.append(Order(f['time'] + datetime.timedelta(minutes=delta), f['res'], self.destination))
        self.dataSource.append(tem_re)


if __name__ == "__main__":
    instance = OrderGenerator(restaurants, destinations, foods, day=1e3)
    instance.csv_encoder()