class Environment1:
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        #日期索引对应的买股票价格
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history  # obs

    def step(self, act):
        reward = 0

        # act = 0: stay, 1: buy, 2: sell
        if act == 1:
            #把买的时候的价格放入到position中
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2:  # sell
            #之前没买过的时候不能卖
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                # 为了简化问题，卖股票的时候一次性把之前买的股票全部卖了
                for p in self.positions:
                    #收益等于卖的时候的价格减去买时候的价格
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t - 1), :]['Close'])

        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done