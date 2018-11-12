import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import datetime
start = datetime.datetime(1976, 6, 1)
end = datetime.datetime(2018, 8, 31)

# quarterly data
gdp = web.DataReader('A191RL1Q225SBEA', 'fred', start, end)
np.median(gdp)

# monthly covariates
sample = web.DataReader(
    [
    'UNRATE', 'LNS14000002', 'CIVPART', 'LNS11300060', 'LNU01300002',
    'UNEMPLOY', 'GS5', 'BAA10YM', 'AAA10YM', 'GS1', 'GS5',
    'FEDFUNDS', 'LRUN64TTUSM156N', 'LNS14000031', 'GS10', 'GS2',
    ], 'fred', start, end)
# daily data
nasdaq = web.DataReader('NASDAQCOM', 'fred', start, end)

nasdaq['month'] = nasdaq.index.month
nasdaq['year'] = nasdaq.index.year
mu = nasdaq.groupby(['year', 'month']).aggregate(np.mean)
sigma = nasdaq.groupby(['year', 'month']).aggregate(np.std)

mon_nasdaq = mu.join(sigma, lsuffix = '_mean', rsuffix = '_std')
mon_nasdaq['nasdaq_vol'] = mon_nasdaq['NASDAQCOM_std']/mon_nasdaq['NASDAQCOM_mean'] * 100
mon_nasdaq.describe()
mon_nasdaq.index = pd.to_datetime(mon_nasdaq.index.map(lambda x: "-".join([str(x[0]), str(x[1]), "1"])))
sample = sample.join(mon_nasdaq['nasdaq_vol'])

# data preprocessing
data = sample.join(gdp.applymap(lambda x: 1 if x < 0  else 0)) # define stressed as gdp growth rate less than 0%
# we need to do this because gdp growth rate is quarterly data
mon_gdp = data['A191RL1Q225SBEA'].values
mon_gdp[0] = 0

# impute quarterly gdp growth into monthly level
import math
for i in range(1, len(mon_gdp)):
    if math.isnan(mon_gdp[i]):
        mon_gdp[i] = mon_gdp[i-1]

data['gdp_growth'] = mon_gdp
data.drop('A191RL1Q225SBEA', axis = 1, inplace = True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
df = np.concatenate(
    [
        scaler.fit_transform(data.drop('gdp_growth', axis = 1).values.astype('float32')),
        np.array([data.gdp_growth.values.astype('float32')]).T
    ], axis = 1
)

def create_datasets(data, lookback):
    sample, target = [], []
    for i in range(len(data) - lookback - 1):
        d = data[i:(i + lookback - 1)]
        sample.append(d)
        target.append(data[i + lookback, 17])
    return np.array(sample), np.array(target)

from keras import layers
from keras import models
from keras import optimizers

out = []

for k in [2, 6, 12, 24, 36, 48]:
    for l in [2, 4, 8, 16, 32, 64]:
        print('running at %d and %d' % (k, l) )
        sample, target = create_datasets(df, k)
        ix = int(np.ceil(0.8 * len(df)))
        sample_train, target_train = sample[0:ix], target[0:ix]
        sample_test, target_test = sample[ix:], target[ix:]

        model = models.Sequential()
        model.add(layers.LSTM(l, input_shape = (sample_train.shape[1], sample_train.shape[-1])))
        model.add(layers.Dense(1, activation = "sigmoid"))
        model.compile(optimizer = optimizers.RMSprop(), loss = "binary_crossentropy", metrics = ['mae'])

        hist = model.fit(sample_train, target_train, epochs = 100, batch_size = 1, verbose = 2, 
                validation_data = (sample_test, target_test))
        out.append([k, l, hist.history['mean_absolute_error'][99], hist.history['val_mean_absolute_error'][99]])

pd.DataFrame(out, columns = ['k', 'l', 'train_mae', 'val_mae']).to_csv('~/Dropbox/notebooks/recession_cv_res.csv', index = False)
