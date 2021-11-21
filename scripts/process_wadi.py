import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret


# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def main():

    train = pd.read_csv('./WADI_14days.csv', index_col=0)
    test = pd.read_csv('./WADI_attackdata_labelled.csv', index_col=0)
    

    train = train.iloc[:, 2:]
    test = test.iloc[:, 3:]


    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())


    train_labels = np.zeros(len(train))
    test_labels = test.attack

    # train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    cols = [x[46:] for x in train.columns] # remove column name prefixes
    train.columns = cols
    test.columns = cols


    x_train, x_test = norm(train.values, test.values)


    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]



    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)


    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    train_df = train_df.iloc[2160:]

    train_df.to_csv('./train.csv')
    test_df.to_csv('./test.csv')

    f = open('./list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
