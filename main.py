from itertools import product
from run import run_experiment

if __name__ == "__main__":
    cities = [3]#,4,5]
    vehicles = [1]#,2,3]
    max_iter = 10
    num_layers = [1]#,2,3]
    is_warm_start = [True]#, False]
    sub_folder = "experiment",
    exp_name = "experiment_"

    sim = list(product(cities, vehicles, num_layers, is_warm_start))
    total = len(sim)
    num_exp = 0
    for city, vehicle, layer, ws in sim:
        num_exp = str(num_exp + 1).zfill(total)
        
        exp_name = f"{exp_name}_{num_exp}"

        run_experiment(
            C = city,
            V = vehicle,
            max_iter = max_iter,
            num_layer = layer,
            is_warm_start = ws,
            sub_folder = sub_folder,
            exp_name = exp_name
        )