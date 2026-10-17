from exp_cutoff_minimo import main as cutoff_main
from exp_recursos import main as recurso_main

cutoff_main(cities=[3, 4, 5], vehicles=[1, 2, 3])
recurso_main(from_csv="cutoff_minimo.csv")