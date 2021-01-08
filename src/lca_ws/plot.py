import matplotlib.pyplot as plt
import pandas as pd

if __name__ =="__main__":
    iters = [i for i in range(1000)]
    df05 = pd.read_csv('log/iteration1.csv')
    df02 = pd.read_csv('log/iteration2.csv')
    df08 = pd.read_csv('log/iteration3.csv')
    df_rd = pd.read_csv('log/iteration_rd.csv')
    scores05 = df05['score'].values.tolist()
    scores02 = df02['score'].values.tolist()
    scores08 = df08['score'].values.tolist()
    scores_rd = df_rd['score'].values.tolist()
    plt.figure(figsize=(15,8))

    plt.subplot(3,1,1)
    plt.plot(iters, scores02)
    plt.plot(iters, scores_rd, linewidth=2, color='r')
    # plt.xlabel('iteration')
    plt.ylabel('score')
    plt.title(r"Policy Improvement $\epsilon$=0.2")
    plt.legend(labels = ['TDL Policy Improvement', 'Random Policy Improvement'], loc = 'lower left')

    plt.subplot(3,1,2)
    plt.plot(iters, scores05)
    plt.plot(iters, scores_rd, linewidth=2, color='r')
    # plt.xlabel('iteration')
    plt.ylabel('score')
    plt.title(r"$\epsilon$=0.5")
    plt.legend(labels = ['TDL Policy Improvement', 'Random Policy Improvement'], loc = 'lower left')

    plt.subplot(3,1,3)
    plt.plot(iters, scores08)
    plt.plot(iters, scores_rd, linewidth=2, color='r')
    plt.xlabel('iteration of improvement')
    plt.ylabel('score')
    plt.title(r"$\epsilon$=0.8")
    plt.legend(labels = ['TDL Policy Improvement', 'Random Policy Improvement'], loc = 'lower left')


    plt.savefig("020508.png", bbox_inches = "tight")
    plt.show()