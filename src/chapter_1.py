import math

from init_array import get_t_array
import matplotlib.pyplot as plt


def estimates_starting_empirical_moments(t_array, k):
    """
    :param t_array: массив наблюдейний
    :param k: номер момента
    :return: оценка начачального к-ого эмпирического момента
    """
    n = len(t_array)
    return sum(map(lambda x: x ** k, t_array)) / n


def point_estimates_4_sample_starting_moments(t_array):
    """
    :param t_array: массив наблюдейний
    :return: кортеж оценок начальных эмпирических моментов
    """
    return (
        estimates_starting_empirical_moments(t_array, 1),
        estimates_starting_empirical_moments(t_array, 2),
        estimates_starting_empirical_moments(t_array, 3),
        estimates_starting_empirical_moments(t_array, 4)
    )


def estimates_first_4_central_moments(moments):
    """
    :param moments: оценки начальных эмпирических моментов
    :return: кортеж оценок первых четырех центральных моментов
    """
    return (
        moments[0],
        moments[1] - moments[0] ** 2,
        moments[2] - 3 * moments[0] * moments[1] + 2 * moments[0] ** 3,
        moments[3] - 4 * moments[0] * moments[2] + 6 * moments[1] * moments[0] ** 2 - 3 * moments[0] ** 4
    )


def skewness(mu):
    """
    :param mu_3: оценка 3-го центрального момента
    :param mu_2: оценка 2-го центрального момента
    :return: коэффициент ассимметрии
    """
    return mu[2] / mu[1] ** (3 / 2)


def excess(mu):
    """
    :param mu_4: оценка 4-го центрального момента
    :param mu_2: оценка 2-го центрального момента
    :return: коэффициент островершинности
    """
    return mu[3] / mu[1] ** 2 - 3


def estimates_4_unbiased_center_moments(t_array, mu):
    """
    :param t_array: массив наблюдейний
    :param mu: кортеж оценок первых четырех центральных моментов
    :return: кортеж оценок первых четырех несмещенных центральных моментов
    """
    n = len(t_array)
    return (
        mu[0],
        sum(map(lambda x: (x - mu[0]) ** 2, t_array)) / (n - 1),
        n ** 2 / ((n - 1) * (n - 2)) * mu[2],
        (n * (n ** 2 - 2 * n + 3) * mu[3] - 3 * n * (2 * n - 3) * mu[1] ** 2) / ((n - 1) * (n - 2) * (n - 3))
    )


def unbiased_skewness(sk, n):
    """
    :param sk: коэффициент ассимметрии
    :param n: колличество наблюдений
    :return: несмещенный коэффициент ассимметрии
    """
    return (n * (n - 1)) ** (1 / 2) / (n - 2) * sk


def unbiased_excess(ex, n):
    """
    :param ex: коэффициент островершинности
    :param n: колличество наблюдений
    :return: несмещенный коэффициент островершинности
    """
    return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * ex + 6)


def get_empirical_distribution_function(t_array):
    """
    :param t_array: массив наблюдейний
    :return: эмпирическую функцию от t
    """
    sort_t_array = sorted(t_array)
    min_t = sort_t_array[0]
    max_t = sort_t_array[-1]
    n = len(t_array)

    def func(t):
        """
        :param t: измерение
        :return: значение эмпирической функции
        """
        if t <= min_t:
            return 0
        if t > max_t:
            return 1
        i = 0
        while i < n:
            if sort_t_array[i] > t:
                break
            i += 1
        return i / n

    return func


def coefficient_aging(moments):
    return (
        moments[0],
        moments[1] / 2,
        moments[2] / 6,
        moments[3] / 24
    )


def is_aging(coef_aging):
    eq_1 = coef_aging[2] * coef_aging[0] - coef_aging[1] ** 2
    eq_2 = coef_aging[3] * coef_aging[1] - coef_aging[2] ** 2
    print("M_3 * M_1 - M_2 ^ 2 = %f\nM_4 * M_2 - M_3 ^ 2 = %f" % (eq_1, eq_2))
    return eq_1 <= 0 and eq_2 <= 0


if __name__ == '__main__':
    t_array = get_t_array()
    n = len(t_array)
    m = point_estimates_4_sample_starting_moments(t_array=t_array)
    print("<------------------------------------------------------------------------------>")
    print("Результаты расчета первых четырех начальных моментов")
    print("m_1 = %f\nm_2 = %f\nm_3 = %f\nm_4 = %f" % m)
    print("<------------------------------------------------------------------------------>")
    mu = estimates_first_4_central_moments(moments=m)
    sk = skewness(mu)
    ex = excess(mu)
    mu_H = estimates_4_unbiased_center_moments(t_array=t_array, mu=mu)
    sk_H = unbiased_skewness(sk=sk, n=n)
    ex_H = unbiased_excess(ex=ex, n=n)
    delta_h = mu_H[1] ** (1 / 2)
    print("Результаты расчета первых четырех центральных несмещенных моментов")
    print("mu_1 = %f\nmu_2_H = %f\ndelta_H = %f\nSk_H = %f\nEx_H = %f" % (m[0], mu_H[1], delta_h, sk_H, ex_H))
    print("<------------------------------------------------------------------------------>")
    coef_aging = coefficient_aging(m)
    print("M_1 = %f\nM_2 = %f\nM_3 = %f\nM_4 = %f" % coef_aging)
    if is_aging(coef_aging):
        print("Критерий выполнен, распределение относистся к \"стареющим\"")
        print("<------------------------------------------------------------------------------>")
    else:
        print("Критерий невыполнен, распределение не относистся к \"стареющим\"")
        print("<------------------------------------------------------------------------------>")
    f_e = get_empirical_distribution_function(t_array)
    t_start = 0
    t_end = int(max(t_array)) + 50
    t = [i for i in range(t_start, t_end)]
    f_e_t = [f_e(i) for i in range(t_start, t_end)]
    plt.plot(t, f_e_t)
    plt.show()
