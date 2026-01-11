#!/usr/bin/env python3
"""
This module combines five different plots into a single figure 
using a variable grid layout.
"""

import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """
    Plots a cubic curve, a scatter plot, two decay graphs, and a histogram.
    """
    x_cubic = np.arange(0, 11)
    y_cubic = x_cubic ** 3

    mean, cov = [69, 180], [[15, 8], [8, 15]]
    np.random.seed(5)
    x_scatter, y_scatter = np.random.multivariate_normal(mean, cov, 2000).T

    x_log = np.arange(0, 28651, 5730)
    y_log = np.exp((np.log(0.5) / 5730) * x_log)

    x_multi = np.arange(0, 21000, 1000)
    y_c14 = np.exp((np.log(0.5) / 5730) * x_multi)
    y_ra226 = np.exp((np.log(0.5) / 1600) * x_multi)

    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(10, 8))
    plt.suptitle('All in One')

    plt.subplot(3, 2, 1)
    plt.plot(x_cubic, y_cubic, color='red')

    plt.subplot(3, 2, 2)
    plt.scatter(x_scatter, y_scatter, color='magenta', s=5)
    plt.title("Men's Height vs Weight", fontsize='x-small')
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')

    plt.subplot(3, 2, 3)
    plt.plot(x_log, y_log)
    plt.yscale('log')
    plt.title("Exponential Decay of C-14", fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')

    plt.subplot(3, 2, 4)
    plt.plot(x_multi, y_c14, 'r--', label='C-14')
    plt.plot(x_multi, y_ra226, 'g-', label='Ra-226')
    plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.legend(fontsize='x-small')

    plt.subplot(3, 1, 3)
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.title('Project A', fontsize='x-small')
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()
