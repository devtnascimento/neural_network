import numpy as np
from sklearn.neural_network import MLPRegressor
from models import gen_model, plot
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for t in range(1, 6):
    print('Carregando Arquivo de teste')
    arquivo = np.load(f'teste{t}.npy')
    x = arquivo[0]
    y = np.ravel(arquivo[1])

    n = 3

    results = []
    for _ in range(n):
        model = gen_model()
        errors = np.zeros(10, dtype=np.float64)
        mse = 0

        best_regr = None
        best_y_est = None
        for i in range(9):
            regr = MLPRegressor(
                hidden_layer_sizes=model["hidden_layer_size"],
                max_iter=model["max_iter"],
                activation=model["activation"],
                solver=model["solver"],
                learning_rate = 'constant',
                n_iter_no_change=model["n_iter_no_change"]
            )
            print('Treinando RNA')
            fitted_regr = regr.fit(x, y)
            print('Preditor')
            new_y_est = fitted_regr.predict(x)
            if not best_regr:
                best_regr = fitted_regr
                best_y_est = new_y_est
            new_mse = mean_squared_error(y, new_y_est)
            if new_mse < mse:
                mse = new_mse
                best_regr = fitted_regr
                best_y_est = new_y_est
            
            errors[i] = new_mse
            
        results.append({"regr": best_regr, "y_est": best_y_est, "model": model, "mean_error": errors.mean(), "std_error": errors.std()})


    sorted_results = sorted(results, key=lambda x: x["mean_error"])

    i = 0
    for result in sorted_results:
        print(result)
        if i == 0:
            plot(t, x, y, result)
        i += 1



