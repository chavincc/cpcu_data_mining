import sklearn, pandas as pd, numpy as np, math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# read data
raw_data = pd.read_excel('datasetA1.xlsx')
row_count = len(raw_data.index)

revenue = raw_data['revenue']

# sanitize features
sanitized_features = raw_data.drop('revenue', axis=1)
sanitized_features['smoker']= pd.Series(np.where(sanitized_features['smoker'].values == 'yes', 1, 0),sanitized_features.index)

sanitized_sex = pd.get_dummies(sanitized_features['sex'])
sanitized_features = sanitized_features.join(sanitized_sex)
sanitized_features = sanitized_features.drop('sex', axis=1)

sanitized_region = pd.get_dummies((sanitized_features['region']))
sanitized_features = sanitized_features.join(sanitized_region)
sanitized_features = sanitized_features.drop('region', axis=1)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    sanitized_features,
    revenue,
    test_size=0.2,
    random_state=1234
)

# linear regression (normal)
rl = LinearRegression()
rl.fit(X_train, y_train)
y_pred = rl.predict(X_test)

# score normal model
rl_r2_score = rl.score(X_test, y_test)
rl_mse_score = mean_squared_error(y_test, y_pred)
print("normal model R^2: {}".format(rl_r2_score))
print("normal model R: {}".format(math.sqrt(rl_r2_score)))
print("normal model mse: {}".format(rl_mse_score))

# linear regression (logged revenue)
y_test_logged = y_test.apply(np.log)
y_train_logged = y_train.apply(np.log)
rl_log = LinearRegression()
rl_log.fit(X_train, y_train_logged)
y_pred_logged = rl_log.predict(X_test)

# score logged model
rl_log_r2_score = rl_log.score(X_test, y_test_logged)
rl_log_mse_score = mean_squared_error(y_test_logged, y_pred_logged)
print("logged model R^2: {}".format(rl_log_r2_score))
print("logged model R: {}".format(math.sqrt(rl_log_r2_score)))
print("logged model mse: {}".format(rl_log_mse_score))
