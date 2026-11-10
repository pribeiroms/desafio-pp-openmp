/*
 * Exercício Final - Análise de Dados com OpenMP
 * Coeficiente de Variação para dados agrupados em intervalos de classe
 *
 * Compilar:  gcc -O2 -fopenmp cv_openmp.c -o cv_openmp -lm
 * Executar:  ./cv_openmp dados.txt            (dados do exercício, N = 1000)
 *            ./cv_openmp --bench 10000000     (teste de desempenho com N grande)
 *
 * Variável de ambiente opcional: OMP_NUM_THREADS=4 ./cv_openmp dados.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define AMPLITUDE_ESTATURA 8.0   /* cm */
#define AMPLITUDE_PESO     4.0   /* kg */
#define REPETICOES_N_PEQUENO 2000 /* N = 1000 roda em microssegundos: repetimos para medir */

typedef struct {
    double lim_inf;   /* limite inferior da 1ª classe        */
    double h;         /* amplitude do intervalo de classe    */
    int    k;         /* número de classes                   */
    int   *freq;      /* f_i: frequência de cada classe      */
    double media;     /* µ                                   */
    double desvio;    /* σ                                   */
    double cv;        /* CV (%)                              */
} Resultado;

/* ------------------------------------------------------------------ */
/* 1) Mínimo e máximo: laço sobre os N dados                          */
/* ------------------------------------------------------------------ */
static void min_max(const double *v, long n, double *mn, double *mx, int paralelo)
{
    double lo = DBL_MAX, hi = -DBL_MAX;
    #pragma omp parallel for if(paralelo) reduction(min:lo) reduction(max:hi)
    for (long i = 0; i < n; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    *mn = lo;
    *mx = hi;
}

/* ------------------------------------------------------------------ */
/* 2) Distribuição de frequências: laço sobre os N dados              */
/*    Cada thread conta numa cópia privada de freq[] e no final o     */
/*    OpenMP soma as cópias (redução de vetor, OpenMP >= 4.5).        */
/* ------------------------------------------------------------------ */
static void frequencias(const double *v, long n, double lim_inf, double h,
                        int k, int *freq, int paralelo)
{
    memset(freq, 0, (size_t)k * sizeof(int));
    #pragma omp parallel for if(paralelo) reduction(+:freq[:k])
    for (long i = 0; i < n; i++) {
        int c = (int)((v[i] - lim_inf) / h);
        if (c >= k) c = k - 1;          /* o máximo fica na última classe */
        freq[c]++;
    }
}

/* ------------------------------------------------------------------ */
/* 3) Média ponderada: µ = Σ(x_i · f_i) / N                           */
/* ------------------------------------------------------------------ */
static double media_ponderada(const int *freq, int k, double lim_inf, double h,
                              long *n_total, int paralelo)
{
    double soma_xf = 0.0;
    long   soma_f  = 0;
    #pragma omp parallel for if(paralelo) reduction(+:soma_xf, soma_f)
    for (int i = 0; i < k; i++) {
        double xi = lim_inf + i * h + h / 2.0;   /* ponto médio */
        soma_xf += xi * freq[i];
        soma_f  += freq[i];
    }
    *n_total = soma_f;
    return soma_xf / (double)soma_f;
}

/* ------------------------------------------------------------------ */
/* 4) Desvio padrão populacional: σ = sqrt( Σ f_i (x_i - µ)² / N )    */
/* ------------------------------------------------------------------ */
static double desvio_padrao(const int *freq, int k, double lim_inf, double h,
                            double mu, long n, int paralelo)
{
    double soma = 0.0;
    #pragma omp parallel for if(paralelo) reduction(+:soma)
    for (int i = 0; i < k; i++) {
        double xi = lim_inf + i * h + h / 2.0;
        double d  = xi - mu;
        soma += freq[i] * d * d;
    }
    return sqrt(soma / (double)n);
}

/* ------------------------------------------------------------------ */
/* Pipeline completo para uma variável                                */
/* ------------------------------------------------------------------ */
static void calcular(const double *v, long n, double h, Resultado *r, int paralelo)
{
    double mn, mx;
    min_max(v, n, &mn, &mx, paralelo);

    r->h       = h;
    r->lim_inf = floor(mn);                          /* 1ª classe começa no piso do mínimo */
    r->k       = (int)floor((mx - r->lim_inf) / h) + 1;
    r->freq    = realloc(r->freq, (size_t)r->k * sizeof(int));

    frequencias(v, n, r->lim_inf, h, r->k, r->freq, paralelo);

    long N;
    r->media  = media_ponderada(r->freq, r->k, r->lim_inf, h, &N, paralelo);
    r->desvio = desvio_padrao(r->freq, r->k, r->lim_inf, h, r->media, N, paralelo);
    r->cv     = (r->desvio / r->media) * 100.0;
}

static void imprimir_tabela(const char *nome, const char *un, const Resultado *r)
{
    printf("\n=== %s (intervalo de classe = %.0f %s) ===\n", nome, r->h, un);
    printf("  Classe              x_i       f_i\n");
    for (int i = 0; i < r->k; i++) {
        double a = r->lim_inf + i * r->h;
        printf("  [%6.1f ; %6.1f)  %7.1f  %8d\n", a, a + r->h, a + r->h / 2.0, r->freq[i]);
    }
    printf("  Media (mu)        = %.4f %s\n", r->media, un);
    printf("  Desvio padrao (s) = %.4f %s\n", r->desvio, un);
    printf("  Coef. de variacao = %.4f %%\n", r->cv);
}

/* Mede o tempo médio de 'reps' execuções do cálculo das duas variáveis */
static double medir(const double *alt, const double *pes, long n, int reps, int paralelo)
{
    Resultado a = {0}, p = {0};
    calcular(alt, n, AMPLITUDE_ESTATURA, &a, paralelo);   /* aquecimento */
    double t0 = omp_get_wtime();
    for (int r = 0; r < reps; r++) {
        calcular(alt, n, AMPLITUDE_ESTATURA, &a, paralelo);
        calcular(pes, n, AMPLITUDE_PESO,     &p, paralelo);
    }
    double t = (omp_get_wtime() - t0) / reps;
    free(a.freq);
    free(p.freq);
    return t;
}

static void analise_desempenho(const double *alt, const double *pes, long n, int reps)
{
    int max_t = omp_get_max_threads();
    double t_seq = medir(alt, pes, n, reps, 0);

    printf("\n=== Analise de desempenho (N = %ld, media de %d execucoes) ===\n", n, reps);
    printf("  Versao        Threads   Tempo (ms)    Speedup   Eficiencia\n");
    printf("  Sequencial    %7d   %10.4f   %8.2fx   %9.1f%%\n", 1, t_seq * 1e3, 1.0, 100.0);

    /* testa 1, 2, 4, 8, ... threads e, por último, o máximo da máquina */
    for (int t = 1; ; t = (t * 2 > max_t && t != max_t) ? max_t : t * 2) {
        omp_set_num_threads(t);
        double tp = medir(alt, pes, n, reps, 1);
        double s  = t_seq / tp;
        printf("  Paralela      %7d   %10.4f   %8.2fx   %9.1f%%\n", t, tp * 1e3, s, 100.0 * s / t);
        if (t == max_t) break;
    }
    omp_set_num_threads(max_t);
}

/* Gera dados sintéticos (só para o modo --bench) */
static double normal(double mu, double sigma)
{
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return mu + sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char **argv)
{
    double *alt = NULL, *pes = NULL;
    long n = 0;
    int reps;

    if (argc >= 3 && strcmp(argv[1], "--bench") == 0) {
        n = atol(argv[2]);
        alt = malloc(n * sizeof(double));
        pes = malloc(n * sizeof(double));
        srand(42);
        for (long i = 0; i < n; i++) {
            alt[i] = normal(170.0, 9.0);
            pes[i] = normal(70.0, 12.0);
        }
        printf("Modo benchmark: %ld pessoas geradas em memoria.\n", n);
        reps = 5;
    } else {
        const char *arq = (argc >= 2) ? argv[1] : "dados.txt";
        FILE *f = fopen(arq, "r");
        if (!f) { perror(arq); return 1; }

        long cap = 1024;
        alt = malloc(cap * sizeof(double));
        pes = malloc(cap * sizeof(double));
        while (fscanf(f, "%lf %lf", &alt[n], &pes[n]) == 2) {
            if (++n == cap) {
                cap *= 2;
                alt = realloc(alt, cap * sizeof(double));
                pes = realloc(pes, cap * sizeof(double));
            }
        }
        fclose(f);
        printf("Arquivo '%s': %ld pessoas lidas.\n", arq, n);
        reps = REPETICOES_N_PEQUENO;
    }

    printf("Threads disponiveis (OpenMP): %d\n", omp_get_max_threads());

    Resultado ra = {0}, rp = {0};
    calcular(alt, n, AMPLITUDE_ESTATURA, &ra, 1);
    calcular(pes, n, AMPLITUDE_PESO,     &rp, 1);

    imprimir_tabela("ESTATURA", "cm", &ra);
    imprimir_tabela("PESO",     "kg", &rp);

    printf("\n>>> CV estatura = %.2f %%\n", ra.cv);
    printf(">>> CV peso     = %.2f %%\n", rp.cv);

    analise_desempenho(alt, pes, n, reps);

    free(ra.freq); free(rp.freq);
    free(alt);     free(pes);
    return 0;
}
