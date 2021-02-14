import math
import numpy

from scipy.optimize import fsolve
from sympy import symbols, gamma, exp, integrate, plot, Array, summation, Heaviside, Abs, init_printing, sqrt, pi, \
    lambdify, pretty

from chapter_1 import point_estimates_4_sample_starting_moments, estimates_first_4_central_moments, \
    estimates_4_unbiased_center_moments
from init_array import get_t_array


def get_gamma_function_parameters(T, D):
    """
    :param T: мат ожидание
    :param D: дисперсия
    :return: кортеж - параметр формы и параметр масштаба
    """
    return (
        T ** 2 / D,
        T / D
    )


def density_of_probability_of_operating_time_failure(a_param, lambda_param):
    """
    :param a_param: параметр формы
    :param lambda_param: параметр масштаба
    :return: кортеж - символьная функция плотности вероятности и аргумент этой функции
    """
    t = symbols("t")
    return lambda_param ** a_param / gamma(a_param) * t ** (a_param - 1) * exp(-lambda_param * t), t


def get_gamma_distribution(density_of_probability, t_arg):
    """
    :param density_of_probability: символьная функция плотности вероятности
    :param t_arg: аргумент символьной функции плотности вероятности
    :return: символьная функция гамма распределения
    """
    return integrate(density_of_probability, t_arg)


def plot_function_overlay(empirical_distribution_function, empirical_arg,
                          gamma_distribution_function, gamma_arg, bound):
    """
    :param empirical_distribution_function: империческая функция распределения - символьная функция
    :param empirical_arg: аргумент символьной функции империческая функция распределениz
    :param gamma_distribution_function: функция гамма распределения - символьная функция
    :param gamma_arg: аргумент символьной гамма функции распределения
    :param bound: кортеж -  границы построения графика
    :return: обьект графика
    """
    p1 = plot(gamma_distribution_function, (gamma_arg, bound[0], bound[1]), show=False)
    p2 = plot(empirical_distribution_function, (empirical_arg, bound[0], bound[1]), show=False)
    p2[0].line_color = 'red'
    p1.append(p2[0])
    return p1


def approximated_gamma_skewness_excess(a):
    """
    :param a: параметр формы
    :return: значения моментов аппроксимирующего гамма-распределения
    """
    return (
        2 / a ** (1 / 2),
        6 / a
    )


def get_symbolic_empirical_distribution_function(t_array):
    """
    Возвращает символьное представление эмпирической функции
    :param t_array: исходные измерения
    :return: эмпирическая функция распределения
    """
    ar = Array(t_array)
    n = len(t_array)
    i, x_arg = symbols("i x", integer=True)
    return (
        summation(Heaviside(x_arg - ar[i]), (i, 0, n - 1)) / n,
        x_arg)


def criterion_kolmogorov(empirical, theoretical, arg, n, bound, alfa):
    """
    :param empirical: символьная функция имперического распределения
    :param theoretical: символьная функция теоретического распределения
    :param arg: аргумент функций
    :param n: кол-во измерений
    :param bound: границы поиска максимума
    :param alfa: уровень значимости
    :return: кортеж - обьект графика, параметр D, лямбда, процентная точка распределения колмогорова
    """
    y_n = Abs(empirical - theoretical)
    D = max([y_n.subs(arg, _).evalf(5) for _ in range(bound[0], bound[1])])
    return (
        plot(y_n, (arg, bound[0], bound[1]), show=False),
        D,
        D * n ** (1 / 2),
        (-1 / 2 * math.log((1 - alfa) / 2)) ** (1 / 2)
    )


def weibull_params(T, D):
    """
    :param T: мат ожидание
    :param D: дисперсия
    :return: кортеж - параметр формы и параметр масштаба
    """
    a = symbols("a")
    equ = - gamma(1 + 2 / a) / gamma(1 + 1 / a) ** 2 + D / T ** 2 + 1
    lambdify_equ = lambdify(a, equ, modules=['numpy'])
    a_param = fsolve(lambdify_equ, 0.5)
    return (
        a_param[0],
        T / gamma(1 + 1 / a_param[0])
    )


def get_weibull_distribution(a_params, b_params):
    """
    :param a_params: параметр формы
    :param b_params: параметр масштаба
    :return: функция распределения вейбула
    """
    t = symbols("t")
    return (
        1 - exp(-(t / b_params) ** a_params),
        t
    )


def approximated_weibull_skewness_excess(a):
    """
    :param a: параметр формы
    :return: значения моментов аппроксимирующего распределения вейбула
    """
    sk = (2 * gamma(1 + 1 / a) ** 3 - 3 * gamma(1 + 1 / a) * gamma(1 + 2 / a) + gamma(1 + 3 / a)) / (
            gamma(1 + 2 / a) - gamma(1 + 1 / a) ** 2) ** (3 / 2)
    ex = (gamma(1 + 4 / a) - 4 * gamma(1 + 3 / a) * gamma(1 + 1 / a) + 6 * gamma(1 + 2 / a) * gamma(
        1 + 1 / a) ** 2 - 3 * gamma(1 + 1 / a) ** 4) / (gamma(1 + 2 / a) - gamma(1 + 1 / a) ** 2) ** 2 - 3
    return (
        sk,
        ex
    )


def analise_with_gamma_function(t_array, bound, gamma_a, gamma_lambda):
    """
    Тут происходит все необходимое для разделов (2,3).1.1 и (2,3).1.2
    вычисляем функцию гамма распределения, импирическую функцию, строим графики, проверяем критейрий колмогорова
    :param t_array: исходные измерения
    :param bound: границы рассмотрения, вычисления максимумов и тд
    :param gamma_a: параметр формы
    :param gamma_lambda: параметр масштаба
    :return:
    """
    n = len(t_array)
    gamma_density, gamma_arg = density_of_probability_of_operating_time_failure(gamma_a, gamma_lambda)
    gamma_distribution = get_gamma_distribution(gamma_density, gamma_arg)
    empirical_distribution, empirical_arg = get_symbolic_empirical_distribution_function(t_array)
    plot = plot_function_overlay(empirical_distribution, empirical_arg, gamma_distribution, gamma_arg, bound)
    sk, ex = approximated_gamma_skewness_excess(gamma_a)
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("a = %f\nlambda = %f" % (gamma_a, gamma_lambda))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    init_printing()
    print("f(t) = ")
    print(pretty(gamma_density))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Sk = %f\nEx = %f" % (sk, ex))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    theoretical = gamma_distribution.subs(gamma_arg, empirical_arg)
    plot_kolmogorov, d_param, lambda_param, k_point = criterion_kolmogorov(empirical_distribution, theoretical,
                                                                           empirical_arg, n, bound, 0.05)
    print("lambda = %f\nX_a = %f" % (lambda_param, k_point))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n\n")
    plot.show()
    plot_kolmogorov.show()


def analise_with_weibull_function(t_array, bound, a_param, b_param):
    """
     Тут происходит все необходимое для разделов (2,3).2.1 и (2,3).2.2
    вычисляем функцию распределения вейбула, импирическую функцию, строим графики, проверяем критейрий колмогорова
    :param t_array: исходные измерения
    :param bound: границы рассмотрения, вычисления максимумов и тд
    :param a_param: параметр формы
    :param b_param: параметр масштаба
    :return:
    """
    n = len(t_array)
    empirical_distribution, empirical_arg = get_symbolic_empirical_distribution_function(t_array)
    weibull_distribution, weibull_arg = get_weibull_distribution(a_param, b_param)
    plot = plot_function_overlay(empirical_distribution, empirical_arg, weibull_distribution, weibull_arg, bound)

    sk, ex = approximated_weibull_skewness_excess(a_param)
    theoretical = weibull_distribution.subs(weibull_arg, empirical_arg)
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("a = %f\nb = %f" % (a_param, b_param))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Sk = %f\nEx = %f" % (sk, ex))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    plot_kolmogorov, d_param, lambda_param, k_point = criterion_kolmogorov(empirical_distribution, theoretical,
                                                                           empirical_arg, n, bound, 0.05)
    print("lambda = %f\nX_a = %f" % (lambda_param, k_point))
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n\n")
    plot.show()
    plot_kolmogorov.show()


def chapter_2_1(t_array, mu_H, bound):
    """
    :param t_array: исходные измерения
    :param mu_H: кортеж в котором интересует матожидание и дисперсия
    :param bound: границы рассмотрения, вычисления максимумов и тд
    :return:
    """
    gamma_a, gamma_lambda = get_gamma_function_parameters(mu_H[0], mu_H[1])
    analise_with_gamma_function(t_array, bound, gamma_a, gamma_lambda)


def chapter_2_2(t_array, mu_H, bound):
    """
    :param t_array: исходные измерения
    :param mu_H: кортеж в котором интересует матожидание и дисперсия
    :param bound: границы рассмотрения, вычисления максимумов и тд
    :return:
    """
    n = len(t_array)
    a_param, b_param = weibull_params(mu_H[0], mu_H[1])
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("D/T^2 + 1 = %f" % (mu_H[1] / mu_H[0] ** 2 + 1))
    analise_with_weibull_function(t_array, bound, a_param, b_param)


if __name__ == '__main__':
    t_array = get_t_array()
    bound = (0, int(max(t_array)) + 50)
    m = point_estimates_4_sample_starting_moments(t_array=t_array)
    mu = estimates_first_4_central_moments(moments=m)
    mu_H = estimates_4_unbiased_center_moments(t_array=t_array, mu=mu)
    chapter_2_1(t_array, mu_H, bound)
    chapter_2_2(t_array, mu_H, bound)
