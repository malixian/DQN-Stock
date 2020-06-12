from subprocess import check_output

print(check_output(["ls", "./data/stocks"]).decode("utf8"))

import time
import numpy as np
import pandas as pd


# plotly 是一个非常牛逼的可视化神器
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
from environment import Environment1

#init_notebook_mode()



def data_init():
    # 数据调整
    data = pd.read_csv('./data/stocks/abac.us.txt')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    #print(data.index.min(), data.index.max())
    data.head()
    return data

def get_train_test(data):
    date_split = '2016-01-01'
    train = data[:date_split]
    test = data[date_split:]
    #len(train), len(test)
    return train, test


def plot_train_test(train, test, date_split):
    data = [
        # Candlestck 是K线
        Candlestick(x=train.index, open=train['Open'], high=train['High'], low=train['Low'], close=train['Close'],
                    name='train'),
        Candlestick(x=test.index, open=test['Open'], high=test['High'], low=test['Low'], close=test['Close'],
                    name='test')
    ]
    layout = {
        'shapes': [
            {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
             'line': {'color': 'rgb(0,0,0)', 'width': 1}}
        ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left',
             'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right',
             'text': 'train data '}
        ]
    }
    figure = Figure(data=data, layout=layout)
    iplot(figure)


def plot_loss_reward(total_losses, total_rewards):

    figure = tools.make_subplots(rows=1, cols=2, subplot_titles=('loss', 'reward'), print_grid=False)
    figure.append_trace(Scatter(y=total_losses, mode='lines', line=dict(color='skyblue')), 1, 1)
    figure.append_trace(Scatter(y=total_rewards, mode='lines', line=dict(color='orange')), 1, 2)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400, width=900, showlegend=False)
    iplot(figure)

if __name__ == "__main__":
    data = data_init()
    train_data, test_data = get_train_test(data)
    print(train_data.iloc[0, :]['Close'])
    '''
    env = Environment1(train_data)
    print(env.reset())
    for _ in range(3):
        pact = np.random.randint(3)
        print(env.step(pact))
    '''


