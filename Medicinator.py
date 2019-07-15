from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# TODO: import model
classifier = {}

expected_medicine = 'Nitrofurantoine'

patient = {
    'medicatie_is_weefselpenetrerend': [1],
    'leeftijd_cat': [2],
    'geslacht': [0],
    'urinewegklachten': [0],
    'tekenen_van_weefselinvasie': [1],
    'diabetes': [0],
    'zwanger': [0],
    'patient_zit_in_nhg_risicogroep': [0]
}


def input_fn(features, batch_size=1024):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


def comma_to_percentage(number):
    return number * 100

# Because `predictions_object` is an "generator", weird stuff happens here
predictions_object = classifier.predict(input_fn=lambda: input_fn(patient))

for prediction_item, expected_item in predictions_object:
    prediction_output = map(
        comma_to_percentage,
        prediction_item['probabilities']
    )
    prediction_output = list(prediction_output)

# Plot graph
title = 'Expected medicine: {}'.format(expected_medicine)
ind = np.arange(4)  # the x locations for the groups
fig = plt.figure(figsize=(5, 7), num=title)
ax = fig.add_subplot(111)

result_bars = ax.bar(ind, prediction_output, 0.3, color='#777777')

# Set labels on the x/y-axes
ax.set_ylabel('Success Probability (%)')
ax.set_xticks(ind)
ax.set_xticklabels(('Nitrofurantoine', 'Amoxicilline',
                    'Trimethoprim', 'Ciprofloxacine'))

# Set labels above bars
for result_bar in result_bars:
    h = result_bar.get_height()
    ax.text(
        result_bar.get_x() + result_bar.get_width() / 2.,
        1.01 * h,
        '{0:.1f}%'.format(float(h)),  # Label
        ha='center',
        va='bottom'
    )

# Highlight the highest result
best_prediction = np.argmax(prediction_output)
result_bars[best_prediction].set_color('blue')

# Display the graph
plt.title('Urinaltract infection treatment prediction')
plt.show()
