
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <chrono>
#include <map>

#include <cuda.h>

using namespace std;

static void HandleError(cudaError_t err, const char *file, int line)  // проверка на ошибку при операции с device-памятью на host
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

#define N 40


__global__ void calc_Z_cuda(double* g_arr, double* E_arr, double* Z, double T, int count)       // расчет статсуммы
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count)
    {
//        if (index < 1000)   printf("%d\t%lf\n", index, pow(2.7182, -29));

        atomicAdd(Z, g_arr[index] * pow(2.7182, (-1 * ((E_arr[index] + 0.17998) / T))));
    }
}

__global__ void calc_PE_cuda(double* g_arr, double* E_arr, double Z, double* PE, double T, int count)        // расчет массива вероятностей
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count)
    {
//        printf("%d\t%lf\n", index, Z);
        PE[index] = (g_arr[index] * pow(2.7182, (-1 * ((E_arr[index] + 0.17998) / T)))) / Z;
    }
}

__global__ void calc_delta_E(double* E_arr, double* PE, double* delta_E, double* delta_E2, int count)        // расчет delta E и delta E2
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        atomicAdd(delta_E, PE[index] * E_arr[index]);
        atomicAdd(delta_E2, PE[index] * pow(E_arr[index], 2));
    }
}

int get_SP_cores(cudaDeviceProp devProp)        // оптимальные размеры блока под используемую видеокарту
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

int main() {
    srand(time(NULL));

    string gem_file = "gem_true.txt";
    string ge_file = "ge.txt";
    string c_true_file = "c_true.txt";

    double difference = .0;              // обычная средняя разница
    double percent_difference = .0;      // средняя разница в процентах

    vector<double> C_true{}, C_current{};

    ifstream c_true(c_true_file);      // пересчитывать С при перевороте

    if (c_true.fail())
        throw invalid_argument(c_true_file + " not found");

    double tmp, tmp_c;

    while(c_true >> tmp >> tmp_c)
        C_true.push_back(tmp_c);

    c_true.close();
    FILE* diff = fopen("diff.txt", "w");

    // Загрузить dos, убрать столбец M -----------------------------------------------------------------------------------------------------------

    {
//
//    ifstream gem_input(gem_file);
//
//    if (gem_input.fail())
//        throw invalid_argument(gem_file + " not found");
//
//
//    map<double, int> GE;
//    GE.clear();
//
//    int gt; double et; int mt;
//    while(gem_input >> gt >> et >> mt)     // убрать столбец M
//    {
//        if (GE.count(et) == 0)        // если в словаре нет ключа Ei
//            GE[et] = gt;                    // создать с gi
//        else                                // если есть
//            GE[et] += gt;                   // добавить gi
//    }
//
//    gem_input.close();
//    ofstream ge_output(ge_file);
//
//    long long sum = 0;
//    for (pair<double, int> map_elem : GE)
//    {
//        ge_output << map_elem.second << "\t" << map_elem.first << endl;
//        sum += map_elem.second;
//    }
//
////    cout << pow(2, N) << "\t" << sum << endl;
//
//    ge_output.close();

    }

    // Загрузить ge -----------------------------------------------------------------------------------------------------------

    ifstream input(ge_file);

    double tmp0, tmp1;
    int rows = 0;
    while(input >> tmp0 >> tmp1)  rows++;       // читаю число строк из файла
    input.close();

    auto g = new double[rows];
    auto E = new double[rows];

    ifstream input1(ge_file);

    int count = 0;

    while(input1 >> tmp0 >> tmp1)
    {
        g[count] = tmp0;
        E[count] = tmp1;

        count++;
    }

    // Переменные CUDA-----------------------------------------------------------------------------------------

    int interval_num = rows;

    int block_dim = 512;
    int grid_dim_c = ((interval_num - 1) / block_dim) + 1;      // количество блоков для расчета теплоемкости (равно числу данных в файле)

    double Z = .0, *dev_Z;
    double *dev_dg_global, *dev_E_int_global, *dev_PE;

    auto PE = new double[interval_num - 1];                     // массив вероятностей
    double delta_E, delta_E2;
    double *dev_delta_E, *dev_delta_E2;

    HANDLE_ERROR(cudaMalloc((void**)&dev_Z, sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_dg_global, (interval_num - 1) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_E_int_global, (interval_num - 1) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_PE, (interval_num - 1) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_delta_E, sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_delta_E2, sizeof(double)));

    // Цикл-----------------------------------------------------------------------------------------------------

    // Каирская -> (, 36500)

    int skip_rows = 0;      // Сколько строк надо пропустить (нижн граница), все энергии в ge отсортированы по возрастанию map-ом и не повторяются
    int step = 25;          // Шаг по энергиям

    while( percent_difference < 0.1 )         // percent_difference < 0.1         skip_rows < rows           skip_rows < skip_rows + 1
    {
        if (skip_rows >= rows - 100)       // замедление на последних итерациях
            step = 1;

        int skip_rows_curr = skip_rows;

        for (int i = 0; i < skip_rows_curr; i++)        // срезание энергий
            g[rows - 1 - i] = 0;                          // начиная с макс
//            g[i] = 0;                                     // начиная с мин

        input1.close();

//        for (int i = 0; i < rows; i++)
//            cout << g[i] << "\t" << E[i] << endl;

        // Рассчитать теплоемкость для текущего ge файла ----------------------------------------------------------------------------------------------------------------------

        C_current.clear();

        FILE* f_c = fopen("c.txt", "w");

        HANDLE_ERROR(cudaMemcpy(dev_dg_global, g, sizeof(double) * (interval_num - 1), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_E_int_global, E, sizeof(double) * (interval_num - 1), cudaMemcpyHostToDevice));

        count = 0;

        double C = .0;

        for (double T = 0.00001; T < 0.12; T += 0.000001)
        {
            for (int i = 0; i < interval_num - 1; i++)  PE[i] = .0;

            HANDLE_ERROR(cudaMemset(dev_Z, 0, sizeof(double)));
            HANDLE_ERROR(cudaMemset(dev_delta_E, 0, sizeof(double)));
            HANDLE_ERROR(cudaMemset(dev_delta_E2, 0, sizeof(double)));
            HANDLE_ERROR(cudaMemcpy(dev_PE, PE, sizeof(double) * (interval_num - 1), cudaMemcpyHostToDevice));

            calc_Z_cuda <<<grid_dim_c, block_dim>>>(dev_dg_global, dev_E_int_global, dev_Z, T, interval_num - 1);           // Z
            HANDLE_ERROR(cudaMemcpy( &Z, dev_Z, sizeof(double), cudaMemcpyDeviceToHost));

            calc_PE_cuda<<<grid_dim_c, block_dim>>>(dev_dg_global, dev_E_int_global, Z, dev_PE, T, interval_num - 1);       // PE
            HANDLE_ERROR(cudaMemcpy( PE, dev_PE, sizeof(double) * (interval_num - 1), cudaMemcpyDeviceToHost));

            if (Z != 0)
            {
                calc_delta_E<<<grid_dim_c, block_dim>>>(dev_E_int_global, dev_PE,dev_delta_E, dev_delta_E2, interval_num - 1);   // delta_E и delta_E2

                HANDLE_ERROR(cudaMemcpy(&delta_E, dev_delta_E, sizeof(double), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(&delta_E2, dev_delta_E2, sizeof(double), cudaMemcpyDeviceToHost));

                C = (delta_E2 - pow(delta_E, 2)) / pow(T, 2);
            }
            else
                C = 0;      // если CUDA отказывается правильно считать (Z == -inf)

//            cout << "T = " << T << "\tC = " << C << endl;
            fprintf(f_c, "%*.*lf\t%*.*lf\n", 5, 5, T, 5, 5, C);   // запись теплоемкости в файл

            C_current.push_back(round(C * 100000) / 100000);    // массив для сравнения с полными значениями

            T *= 1.05;
            count++;
        }

        fclose(f_c);

        // Найти разницу с эталоном ----------------------------------------------------------------------------------------------------------------------

        double sum = .0, sum_p = .0;
        int local_count = 0;

        int ignore_low_c = 10;      // игнорировать участки с ошибками в начале

        for (int i = ignore_low_c; i < count; i++)
        {
            if (C_true[i] != C_current[i])                                  // учитываются только различающиеся C
            {
                if (C_current[i] < C_true[i])
                {
                    sum += abs(C_true[i] - C_current[i]);
                    sum_p += abs(C_true[i] - C_current[i]) / C_true[i];

                    local_count++;
                }
            }

//            if (C_true[i] != C_current[i])
//                cout << i << " Diff: " <<  C_true[i] << "\t" << C_current[i] << "\t" << (C_true[i] - C_current[i]) / C_true[i] << endl;
//            else
//                cout << i << " " << C_true[i] << "\t" << C_current[i] << "\t" << (C_true[i] - C_current[i]) / C_true[i] << endl;
        }

//        cout << sum << "\t" << local_count << "\t" << sum / local_count << "\t" << sum_p << "\t" << sum_p / local_count << endl;

        if (local_count == 0)
        {
            difference = .0;
            percent_difference = .0;
        }
        else
        {
            difference = sum / local_count;                   // обычная средняя разница
            percent_difference = sum_p / local_count;         // средняя разница в процентах
        }

        cout << skip_rows << "\t" << difference << "\t" << percent_difference << endl;
        fprintf(diff, "%d\t%*.*lf\t%*.*lf\n", skip_rows, 5, 5, difference, 5, 5, percent_difference);   // запись в diff.txt


        skip_rows += step;
    }

    cudaFree(dev_dg_global);
    cudaFree(dev_E_int_global);
    cudaFree(dev_PE);

    delete[] E;
    delete[] g;

    fclose(diff);

    return 0;
}
