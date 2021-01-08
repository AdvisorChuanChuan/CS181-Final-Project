import matplotlib.pyplot as plt

if __name__ == '__main__':
    name_list = ['Greedy', '0.2', '0.5', '0.8']
    success_list = [4,4,3,2]
    overtime_list = [4,7,8,8]
    fail_list = [2,2,1,2]

    x=list(range(len(success_list)))
    total_width, n = 0.8, 3
    width = total_width / n

    plt.figure(figsize=(15,8))
    plt.bar(x, success_list, width=width, label='succeed')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, overtime_list, width=width, label='overtime', tick_label=name_list)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, fail_list, width=width, label='fail')
    plt.title("Orders Outcome")
    plt.legend()
    plt.savefig("ordersNum.png", bbox_inches = "tight")
    plt.show()