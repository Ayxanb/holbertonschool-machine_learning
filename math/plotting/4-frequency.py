#!/usr/bin/env python3
"""
This module provides a function to visualize student performance
on a specific project using a frequency distribution histogram.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades for Project A.
    The distribution is displayed with bins every 10 units,
    ranging from 0 to 100, with black outlines on each bar.
    """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')

    bins = range(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xticks(bins)
    plt.yticks(range(0, 31, 5))
    plt.xlim(0, 100)
    plt.show()
