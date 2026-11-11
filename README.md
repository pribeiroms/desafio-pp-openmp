# desafio-pp-openmp
Cálculo do coeficiente de variação de estatura e peso com dados agrupados em intervalos de classe, paralelizado com OpenMP em C, com comparação de desempenho entre as versões sequencial e paralela. Exercício final de Programação Paralela – IESB.

## Como compilar e executar

```
gcc -O2 -fopenmp cv_openmp.c -o cv_openmp -lm
./cv_openmp dados/dados.txt          # dados do exercício (1000 pessoas)
./cv_openmp --bench 10000000         # teste de desempenho com 10 milhões de registros
```

## Arquivos

- `cv_openmp.c`: código-fonte
- `dados/dados.txt`: conjunto fictício de 1000 pessoas (estatura em cm e peso em kg)
- `relatorio/relatorio_cv_openmp.pdf`: relatório com as partes paralelizadas, os resultados e a análise de desempenho

## Resultados

- CV da estatura: 5,37 %
- CV do peso: 18,08 %
EOF