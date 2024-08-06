import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import os
from pandas import DataFrame as df
import math
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import time
import warnings
warnings.filterwarnings('ignore')

tf.reset_default_graph()

dataFileName = sys.argv[1]
drugInfoFileName = sys.argv[2]
n_H = int(sys.argv[3]) # H : length of the output (number of timepoints to predict)
n = int(sys.argv[4]) #5

data = pd.read_csv(dataFileName, index_col = 0)
data = data.dropna()
original_data = data.copy()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled)
data_scaled.columns = data.columns
data_scaled.index = data.index
temp_data = data_scaled

num_total = len(temp_data)
num_train = int(round(num_total * 0.8))
num_test = num_total - num_train

n_time_in = n * n_H
n_time_out = n_H

drug_info = pd.read_csv(drugInfoFileName)
temp_data_train = temp_data[:num_train]
temp_data_train = temp_data_train.T

temp_data_train_drug = pd.merge(drug_info, temp_data_train, right_index = True, left_on = "gene")
drug_count_df = pd.DataFrame(temp_data_train_drug.groupby('drug').count()['gene'])
drug_count_df.sort_values(by = "gene", ascending = False, inplace = True)
drug_count_df.reset_index(inplace = True, drop = False)

drug_cluster_df = pd.DataFrame()
flag_break = 0
cluster_num = 0
last_cluster_df = pd.DataFrame()

for i in range(len(drug_count_df)) : 
    tmp_arg_data = temp_data_train_drug[temp_data_train_drug['drug'] == drug_count_df['drug'][i]]
    if (len(tmp_arg_data) < 10) :
        last_cluster_df = pd.concat([last_cluster_df, tmp_arg_data], axis = 0)
        if i != (len(drug_count_df)-1) :
            continue
        else :
            tmp_arg_data = last_cluster_df
    tmp_arg_data.set_index('gene', inplace = True, drop = True)
    del tmp_arg_data['drug']
    tmp_arg_data = tmp_arg_data.T
    tmp_arg_data_corr = tmp_arg_data.corr(method = 'spearman')
    tmp_arg_in_drug_list = tmp_arg_data_corr.index.tolist()
    tmp_arg_in_drug_corr_list = []
    for arg in range(len(tmp_arg_in_drug_list)) :
        tmp_arg_row_abs = tmp_arg_data_corr.iloc[arg].abs()
        tmp_cumulative_corr = 1
        for j in range(len(tmp_arg_row_abs)) :
            if tmp_arg_row_abs[j] != 0.0 :
                tmp_cumulative_corr *= tmp_arg_row_abs[j]
        tmp_arg_in_drug_corr_list.append(tmp_cumulative_corr ** (1/len(tmp_arg_row_abs)))
    tmp_arg_in_drug_corr_df = pd.DataFrame({'arg' : tmp_arg_in_drug_list, 'corr' : tmp_arg_in_drug_corr_list})
    tmp_arg_in_drug_corr_df = tmp_arg_in_drug_corr_df.sort_values(by = 'corr', ascending = False)
    del tmp_arg_in_drug_corr_df['corr']
    tmp_arg_in_drug_corr_df['cluster'] = cluster_num
    drug_cluster_df = pd.concat([drug_cluster_df, tmp_arg_in_drug_corr_df], axis = 0)
    cluster_num += 1

num_clusters = cluster_num+1

temp_data = temp_data.T
temp_data = pd.merge(temp_data, drug_cluster_df, left_index = True, right_on = "arg", how = 'right')
temp_data.set_index('arg', inplace = True, drop = True)
temp_data_cluster = temp_data.copy()
del temp_data['cluster']
temp_data = temp_data.T
feature_list = temp_data.columns
num_feature = len(feature_list)

cluster_num = len(temp_data_cluster["cluster"].unique().tolist())
cluster_df_list = []
for i in range(cluster_num) :
	tmp_cluster = pd.DataFrame(temp_data_cluster[temp_data_cluster["cluster"] == i])
	del tmp_cluster["cluster"]
	tmp_cluster = tmp_cluster.T
	cluster_df_list.append(tmp_cluster)

cluster_feature_num_list = []
for i in range(cluster_num) :
	cluster_feature_num_list.append(len(cluster_df_list[i].columns))
	cluster_df_list[i] = cluster_df_list[i].values

tf_X_cluster_list = []
for i in range(len(cluster_df_list)) :
	tf_X_cluster_list.append(tf.placeholder(tf.float32, [None, n_time_in, cluster_feature_num_list[i]])) #None,

tf_Y = tf.placeholder(tf.float32, [n_time_out, num_feature])
phase_nbeats = tf.placeholder(tf.bool, name = "phase_nbeats")

dp_rate = 0.0
n_h1 = 64
n_h2 = 64
n_h3 = 64
n_h4 = 64
n_theta = 32

def cnn_layer(_x, _num_feature, _scope): 
	with tf.variable_scope(_scope):
		cnn_1 = tf.keras.layers.Conv1D(8, 3, strides = 1, padding = 'same', input_shape = (n_time_in, _num_feature), activation = tf.nn.leaky_relu)(_x)
		cnn_1_maxpool = tf.keras.layers.MaxPool1D(pool_size = 1, strides = 1, padding = 'same')(cnn_1)
		return cnn_1_maxpool

def trend_Block(_X, blockNum, _phase) :
	fc1 = tf.keras.layers.Dense(n_h1, activation = "relu", trainable = _phase)(_X)
	fc2 = tf.keras.layers.Dense(n_h2, activation = "relu", trainable = _phase)(fc1)
	fc3 = tf.keras.layers.Dense(n_h3, activation = "relu", trainable = _phase)(fc2)
	fc4 = tf.keras.layers.Dense(n_h4, activation = "relu", trainable = _phase)(fc3)
	theta_f = tf.keras.layers.Dense(n_theta, trainable = _phase)(fc4)
	theta_b = tf.keras.layers.Dense(n_theta, trainable = _phase)(fc4)

	#Backcast
	tmp_row = []
	for i in range(n_theta) :
		tmp_col = []
		for j in range(n_time_in) :
			tmp_col.append(pow((float(j)/n_time_in), i*(2.0/n_theta)))
		tmp_row.append(tmp_col)

	trend_t_backast = tf.constant(tmp_row)
	trend_backast = tf.matmul(theta_b, trend_t_backast)

	#Forecast
	tmp_row = []
	for i in range(n_theta) :
		tmp_col = []
		for j in range(n_time_out) :
			tmp_col.append(pow((float(j)/n_time_out), i*(2.0/n_theta)))
		tmp_row.append(tmp_col)

	trend_t_forecast = tf.constant(tmp_row)
	trend_forecast = tf.matmul(theta_f, trend_t_forecast)

	trend_backast = _X - trend_backast
	return trend_backast, trend_forecast


def seasonal_Block(_X, blockNum, _phase) :
	fc1 = tf.keras.layers.Dense(n_h1, activation = "relu", trainable = _phase)(_X)
	fc2 = tf.keras.layers.Dense(n_h2, activation = "relu", trainable = _phase)(fc1)
	fc3 = tf.keras.layers.Dense(n_h3, activation = "relu", trainable = _phase)(fc2)
	fc4 = tf.keras.layers.Dense(n_h4, activation = "relu", trainable = _phase)(fc3)
	theta_f = tf.keras.layers.Dense(n_theta, trainable = _phase)(fc4)
	theta_b = tf.keras.layers.Dense(n_theta, trainable = _phase)(fc4)

	#Backcast
	tmp_row = []
	for i in range(int(n_theta/2)) :
		tmp_col = []
		for j in range(n_time_in) :
			tmp_val = math.cos(2*math.pi*(float(j)/2 - 1.0) * float(i))
			tmp_col.append(tmp_val)
		tmp_row.append(tmp_col)

	for i in range(int(n_theta/2)) :
		tmp_col = []
		for j in range(n_time_in) :
			tmp_val = math.sin(2*math.pi*(float(j)/2 - 1.0) * float(i))
			tmp_col.append(tmp_val)
		tmp_row.append(tmp_col)

	seasonal_t_backcast = tf.constant(tmp_row)
	seasonal_backast = tf.matmul(theta_b, seasonal_t_backcast)

	#Forecast
	tmp_row = []
	for i in range(int(n_theta/2)) :
		tmp_col = []
		for j in range(n_time_out) :
			tmp_val = math.cos(2*math.pi*(float(j)/2 - 1.0) * float(i))
			tmp_col.append(tmp_val)
		tmp_row.append(tmp_col)

	for i in range(int(n_theta/2)) :
		tmp_col = []
		for j in range(n_time_out) :
			tmp_val = math.sin(2*math.pi*(float(j)/2 - 1.0) * float(i))
			tmp_col.append(tmp_val)
		tmp_row.append(tmp_col)

	seasonal_t_forecast = tf.constant(tmp_row)
	seasonal_forecast = tf.matmul(theta_b, seasonal_t_forecast)

	seasonal_backast = _X - seasonal_backast
	return seasonal_backast, seasonal_forecast


def residual_Block(_X, blockNum, _phase) :
	fc1 = tf.keras.layers.Dense(n_h1, activation = "relu", trainable = _phase)(_X)
	fc2 = tf.keras.layers.Dense(n_h2, activation = "relu", trainable = _phase)(fc1)
	fc3 = tf.keras.layers.Dense(n_h3, activation = "relu", trainable = _phase)(fc2)
	fc4 = tf.keras.layers.Dense(n_h4, activation = "relu", trainable = _phase)(fc3)
	theta_f = tf.keras.layers.Dense(n_theta, trainable = _phase)(fc4)
	theta_b = tf.keras.layers.Dense(n_theta, trainable = _phase)(fc4)

	residual_backast = tf.keras.layers.Dense(n_time_in, trainable = _phase)(theta_b)
	residual_forecast = tf.keras.layers.Dense(n_time_out, trainable = _phase)(theta_f)

	residual_backast = _X - residual_backast
	return residual_backast, residual_forecast

for i in range(len(tf_X_cluster_list)) :
    if i == 0 :
        cluster_cnn_feature = cnn_layer(tf_X_cluster_list[i], cluster_feature_num_list[i], "cnn_cluster_" + str(i))
    else :
        tmp_feature = cnn_layer(tf_X_cluster_list[i], cluster_feature_num_list[i], "cnn_cluster_" + str(i))
        cluster_cnn_feature = tf.concat([cluster_cnn_feature, tmp_feature], 2)

cluster_cnn_feature = tf.keras.layers.Dense(num_feature, trainable = phase_nbeats)(cluster_cnn_feature)

cluster_cnn_feature = tf.reshape(cluster_cnn_feature, [-1, n_time_in])

trend_backast_b1, trend_forecast_b1 = trend_Block(cluster_cnn_feature, "1", phase_nbeats)
seasonal_backast_b1, seasonal_forecast_b1 = seasonal_Block(trend_backast_b1, "1", phase_nbeats)
residual_backast_b1, residual_forecast_b1 = residual_Block(seasonal_backast_b1, "1", phase_nbeats)

pred_forecast = trend_forecast_b1 + seasonal_forecast_b1 + residual_forecast_b1
pred_forecast = tf.reshape(pred_forecast, [-1, num_feature])

learning_rate = 1e-3
num_train_epoch = 3000

nbeats_cost = tf.reduce_mean(tf.abs(pred_forecast - tf_Y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(nbeats_cost)

min_mae = 1000
early_stop = 0

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_train_epoch) :
        i = 0
        avg_mae_loss = 0
        iter = 0
        train_end_idx = i + n_time_in
        test_end_idx = train_end_idx + n_time_out
        while test_end_idx < num_train :
            feed_dict = {tf_Y : temp_data[train_end_idx : test_end_idx], phase_nbeats : True} #tf_X : temp_data[i:train_end_idx],
            for t in range(len(cluster_df_list)) :
                feed_dict[tf_X_cluster_list[t]] = np.expand_dims(cluster_df_list[t][i:train_end_idx], axis = 0)
            _, mae_loss = sess.run([train_op, nbeats_cost], feed_dict = feed_dict)
            avg_mae_loss += mae_loss
            iter += 1
            i += 1
            train_end_idx = i + n_time_in
            test_end_idx = train_end_idx + n_time_out
        avg_mae_loss /= iter
        if epoch % 10 == 0 :
            eval_test_start_idx = num_train
            eval_train_start_idx = eval_test_start_idx - n_time_in
            eval_test_end_idx = eval_test_start_idx + n_time_out
            eval_i = 0
            eval_iter = 0
            while eval_test_end_idx < num_total :
                y_test = temp_data[eval_test_start_idx : eval_test_end_idx]
                feed_dict = {tf_Y : y_test, phase_nbeats : False} #
                for t in range(len(cluster_df_list)) :
                    feed_dict[tf_X_cluster_list[t]] = np.expand_dims(cluster_df_list[t][eval_train_start_idx:eval_test_start_idx], axis = 0)
                test_pred, mae_loss = sess.run([pred_forecast, nbeats_cost], feed_dict = feed_dict)
                original_feature_list = data_scaled.columns.tolist()
                y_test = y_test.T.drop_duplicates().T
                y_test = y_test[original_feature_list]
                test_pred = pd.DataFrame(test_pred)
                test_pred = test_pred.T
                test_pred['arg'] = feature_list
                test_pred.drop_duplicates('arg', keep = 'first', inplace = True)
                test_pred.set_index('arg', inplace = True)
                test_pred = test_pred.T
                test_pred = test_pred[original_feature_list]
                inverse_result = scaler.inverse_transform(test_pred)
                inverse_result = pd.DataFrame(inverse_result)
                inverse_result.columns = original_feature_list
                for feature in original_feature_list :
                    inverse_result.loc[inverse_result[feature] < 0, feature] = 1e-8
                    forecast = inverse_result[feature]
                y_test.reset_index(inplace = True, drop = True)
                if eval_i == 0 :
                    final_inverse_result = inverse_result.copy()
                    final_result = test_pred.copy()
                else :
                    final_inverse_result.loc[len(final_inverse_result)] = inverse_result.loc[9]
                    final_result.loc[len(final_result)] = test_pred.loc[9]
                eval_avg_mae_loss += mae_loss
                eval_iter += 1
                eval_i += 1
                eval_test_start_idx += 1
                eval_train_start_idx = eval_test_start_idx - n_time_in
                eval_test_end_idx = eval_test_start_idx + n_time_out
            eval_avg_mae_loss /= eval_iter
            if min_mae >= eval_avg_mae_loss :
            	min_mae = eval_avg_mae_loss
            	early_stop = 0
            else :
            	early_stop += 1
            if early_stop == 200 : #30
            	break
            print("Epoch: %d, train_loss: %f, test_loss: %f" % (epoch, avg_mae_loss, eval_avg_mae_loss))

saveDir = "./results"
os.makedirs(saveDir, exist_ok = True)
final_inverse_result.to_csv(saveDir + 'final_inverse_result.csv', mode = "w", index = False)
final_result.to_csv(saveDir + 'final_result.csv', mode = "w", index = False)
