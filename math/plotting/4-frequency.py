#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.title('Project A')
    plt.xlabel('Number of Students')
    plt.ylabel('Grades')
    plt.hist(student_grades, bins=10, edgecolor='black')
    plt.show()
