import matplotlib.pyplot as plt
import pandas as pd

if __name__ =="__main__":
    iters = [i for i in range(1000)]
    df = pd.read_csv('log_lca/iteration1.csv')
    df_rd = pd.read_csv('log_lca/iteration_rd.csv')
    scores = df['score'].values.tolist()
    scores_rd = df_rd['score'].values.tolist()
    plt.plot(iters, scores)
    plt.plot(iters, scores_rd)
    plt.show()
