from sklearn.neighbors import KNeighborsRegressor
kneighbor_regression = KNeighborsRegressor(n_neighbors=1)
kneighbor_regression.fit(X_train, y_train)

y_pred_train = kneighbor_regression.predict(X_train)

plt.plot(X_train, y_train, 'o', label="data")
plt.plot(X_train, y_pred_train, 'o', label="prediction")
plt.legend(loc='best')

#y_pred_test = kneighbor_regression.predict(X_test)

#plt.plot(X_test, y_test, 'o', label="data")
#plt.plot(X_test, y_pred_test, 'o', label="prediction")
#plt.legend(loc='best')
