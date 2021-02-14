import math

from scipy.optimize import fsolve

import numpy
from chapter_1 import point_estimates_4_sample_starting_moments
from chapter_2 import analise_with_gamma_function, analise_with_weibull_function
from init_array import get_t_array
from sympy import symbols, Array, summation, lambdify, solve, log


def left_part_equation(t_array, T):
    n = len(t_array)
    return 1 / n * sum(map(lambda x: math.log(x / T), t_array))


def q_function():
    """
    :return: кортеж - символьная функция Q(a) и ее аргумент
    """
    a = symbols("a")
    return (
        1 + 1 / (12 * a) + 1 / (288 * a ** 2) - 139 / (51840 * a ** 3) - 571 / (2488320 * a ** 4),
        a
    )


def g_function():
    """
    :return: кортеж - символьная функция G(a) и ее аргумент
    """
    a = symbols("a")
    q, q_arg = q_function()
    q = q.subs(q_arg, a)
    return (
        -1 + (a - 0.5) / a
        + (
                -1 / (12 * a ** 2)
                - 2 / (288 * a ** 3)
                + 3 * 139 / (51840 * a ** 4)
                + 571 * 4 / (2488320 * a ** 5)
        ) / q,
        a
    )


def get_weibull_param(t_array):
    """
    :param t_array: исходный массив
    :return: кортеж - параметр формы и параметр масштаба
    """
    n = len(t_array)
    a, i = symbols("a i")
    arr = Array(t_array)
    eq = n / a + sum(map(lambda x: math.log(x), t_array)) - n * (
        summation(arr[i] ** a * log(arr[i]), (i, 0, n - 1))) / (summation(arr[i] ** a, (i, 0, n - 1)))
    lambdify_equ = lambdify(a, eq, modules=['numpy'])
    a_param = fsolve(lambdify_equ, 0.5)[0]
    return (
        a_param,
        (sum(map(lambda x: x ** a_param, t_array)) / n) ** (1 / a_param)
    )


def chapter_3_1(t_array, T, bound):
    """
    :param t_array: исходные измерения
    :param T: матожидание
    :param bound: границы рассмотрения, вычисления максимумов и тд
    :return:
    """
    left_part = left_part_equation(t_array, T)
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("left_part = %f" % left_part)
    g, g_arg = g_function()
    g = g - left_part
    roots = solve(g, g_arg)
    print("all roots = " + str(roots))
    a_param = roots[2]
    lambda_param = a_param / T
    analise_with_gamma_function(t_array, bound, a_param, lambda_param)


def chapter_3_2(t_array, bound):
    """
    :param t_array: исходные измерения
    :param T: матожидание
    :param bound: границы рассмотрения, вычисления максимумов и тд
    :return:
    """
    a, b = get_weibull_param(t_array)
    analise_with_weibull_function(t_array, bound, a, b)


if __name__ == "__main__":
    t_array = get_t_array()
    bound = (0, int(max(t_array)) + 50)
    n = len(t_array)
    T = point_estimates_4_sample_starting_moments(t_array=t_array)[0]
    chapter_3_1(t_array, T, bound)
    chapter_3_2(t_array, bound)
