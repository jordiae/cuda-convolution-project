#!/bin/bash

export PATH=/Soft/cuda/9.0.176/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N deviceQuery
# Cambiar el shell
#$ -S /bin/bash

./imfilter.exe img/image_barcelona_128.jpg
./imfilter.exe img/image_barcelona_512.jpg
./imfilter.exe img/image_barcelona_3072.jpg
./imfilter.exe img/image_barcelona_4096.jpg
