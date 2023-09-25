import random
import matplotlib.pyplot as plt
import json

def gen_layers() -> tuple:
    sizes = [2, 4, 6, 16, 32, 64, 128]

    n_layers = random.randint(1, 6)

    hidden_layer_sizes = []
    while n_layers > 0 and len(sizes) > 1:
        layer = random.choice(sizes)
        hidden_layer_sizes.append(layer)
        idx = sizes.index(layer)
        sizes = sizes[:idx + 1]
        n_layers -= 1
    return tuple(hidden_layer_sizes)

model_pool = {
    "max_iter": {200, 400, 800, 1600, 3200, 6400, 12800},
    "activation": {'identity', 'logistic', 'tanh', 'relu'},
    "n_iter_no_change": {50, 100, 1000},
    "hidden_layer_size": gen_layers()
}

def gen_model():

    model = {
        "hidden_layer_size": gen_layers(),
        "max_iter": random.choice(list(model_pool["max_iter"])),
        "activation": random.choice(list(model_pool["activation"])),
        "solver": 'adam',
        "n_iter_no_change": random.choice(list(model_pool["n_iter_no_change"]))
    }

    return model



def plot(t, x, y, result):

    plt.figure(figsize=[14,7])

    str_model = json.dumps(result["model"])

    # plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    # plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(result["regr"].loss_curve_)

    # plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,result["y_est"],linewidth=2)

    with open(f"resultados/result_teste{t}.txt", "w") as file:
        file.write(f"Arquitetura: {str_model}\nMedia: {result['mean_error']}\nDesvio: {result['std_error']}")


    plt.savefig(f'resultados/result_teste{t}.png')

