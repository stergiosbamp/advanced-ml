import pandas as pd
import os
import json


def populated_metrics_df(directory):
    metrics_df = pd.DataFrame()
    
    for filename in os.listdir(directory):
        
        the_file = os.path.join(directory, filename)
        
        with open(the_file) as f:
            data = json.load(f)
            df = pd.DataFrame.from_dict([data], orient='columns')
            df.index = [filename]
            # df['combo'] = filename
            metrics_df = metrics_df.append(df)
    
    return metrics_df


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./results/imbalance')
    df_bal = populated_metrics_df(directory='./results/balance')

    # Get the index of the best performing model for imbalanced experiments
    print("\nFor imbalanced dataset best performing model per metric is:\n")
    print(df_imbal.idxmax())
    
    # Get the index of the best performing sampling + model for balanced experiments
    print("\nFor balanced dataset best performing model per metric is:\n")
    print(df_bal.idxmax())

    print("\nRow metrics are:")
    print(df_imbal, "\n")
    print(df_bal, "\n")
