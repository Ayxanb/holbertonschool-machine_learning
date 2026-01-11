#!/usr/bin/env python3
"""
This module provides a function to generate a stacked bar chart
representing the distribution of fruit quantities among individuals.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar chart of fruit counts (apples, bananas, oranges,
    and peaches) for three specific people: Farrah, Fred, and Felicia.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    plt.figure(figsize=(6.4, 4.8))

    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')

    labels = ['Farrah', 'Fred', 'Felicia']

    plt.bar(labels, fruit[0], label='apples', color='red', width=0.5)
    plt.bar(labels, fruit[1], bottom=np.sum(fruit[:1], axis=0),
            label='bananas', color='yellow', width=0.5)
    plt.bar(labels, fruit[2], bottom=np.sum(fruit[:2], axis=0),
            label='oranges', color='#ff8000', width=0.5)
    plt.bar(labels, fruit[3], bottom=np.sum(fruit[:3], axis=0),
            label='peaches', color='#ffe5b4', width=0.5)

    plt.ylim(0, 80)
    plt.legend()
    plt.show()
