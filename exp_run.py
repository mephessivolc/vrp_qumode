from exp_cutoff_minimo import main as cutoff_main
from exp_recursos import main as recurso_main

cutoff_main(
    cities=[3,4,5], 
    vehicles=[1,2,3], 
    cutoffs=[3,4,5,6,7,8,10,12,14,16,20],
    max_dim=12000,
    max_configs=300000,
    out="cutoff_minimo.csv"
)

recurso_main(
    from_csv="cutoff_minimo.csv"
)