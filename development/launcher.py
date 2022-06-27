import os

inn = [0.1, 0.01, 0.001]
out = [1e-3, 1e-2, 1e-1]

for inner in inn:
    for outer in out:
        os.system('/home/carlos/miniconda3/envs/garage/bin/python /home/carlos/repositorios/garage/development/pruebas_tasksampler_resnet.py --inner_lr {} --outer_lr {}'.format(inner, outer))

