#!/usr/bin/env python3
"""Module that plots multiple graphs in a single figure."""
import numpy as np
import matplotlib.pyplot as plt

# !Note: i got help from AI for this task!

def all_in_one():
    """Plots five graphs using two different grid layouts."""
    
    # --- 1. PREPARE ALL DATA ---
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

    # --- 2. SETUP FIGURE ---
    plt.figure(figsize=(10, 8))
    plt.suptitle('All in One')

    # --- 3. DRAW THE PLOTS ---

    # Plot 1: Top Left (3 rows, 2 columns, position 1)
    plt.subplot(3, 2, 1)
    plt.plot(x_cubic, y_cubic, color='red')

    # Plot 2: Top Right (3 rows, 2 columns, position 2)
    plt.subplot(3, 2, 2)
    plt.scatter(x_scatter, y_scatter, color='magenta', s=5)
    plt.title("Men's Height vs Weight", fontsize='x-small')

    # Plot 3: Middle Left (3 rows, 2 columns, position 3)
    plt.subplot(3, 2, 3)
    plt.plot(x_log, y_log)
    plt.yscale('log') # Makes it a log scale
    plt.title("Exponential Decay of C-14", fontsize='x-small')

    # Plot 4: Middle Right (3 rows, 2 columns, position 4)
    plt.subplot(3, 2, 4)
    plt.plot(x_multi, y_c14, 'r--', label='C-14')
    plt.plot(x_multi, y_ra226, 'g-', label='Ra-226')
    plt.title("Radioactive Elements", fontsize='x-small')
    plt.legend(fontsize='x-small')

    # Plot 5: Bottom Row (3 rows, 1 column, position 3)
    # This spans the whole row because we said '1 column'
    plt.subplot(3, 1, 3)
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.title('Project A', fontsize='x-small')
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')

    plt.tight_layout()
    plt.show()
