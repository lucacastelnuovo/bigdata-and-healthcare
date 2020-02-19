from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Import data set
DATA_URL = 'https://cdn.lucacastelnuovo.nl/files/bigdata/csv/pre_data.csv'
dataframe = pd.read_csv(DATA_URL)

train, test = train_test_split(dataframe, test_size=0.01)

# Remove label from features
train_y = train.pop('medicatie_code')
test_y = test.pop('medicatie_code')

# Define feauture data type
feature_columns = [
    feature_column.numeric_column("leeftijd_cat"),
    feature_column.numeric_column("geslacht"),
    feature_column.numeric_column("urinewegklachten"),
    feature_column.numeric_column("tekenen_van_weefselinvasie"),
    feature_column.numeric_column("diabetes"),
    feature_column.numeric_column("zwanger"),
    feature_column.numeric_column("patient_zit_in_nhg_risicogroep")
]


def input_fn(features, labels, shuffle=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if shuffle:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

train_ds = input_fn(train, train_y)
test_ds = input_fn(test, test_y, shuffle=False)

##########################
# Difference in learning #
##########################

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[40, 10],  # Two hidden layers of 10 nodes each.
    n_classes=4  # The model must choose between 4 medicines.
)

classifier.train(
    input_fn=lambda: input_fn(train, train_y),
    steps=5000
)

# TODO: add more epochs
# dataset.batch().repeat(200)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, shuffle=False)
)

print("Accuracy: {accuracy:0.3f}\n".format(**eval_result))

# TODO: SAVE MODEL


def input_fn(features, batch_size=1024):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


def comma_to_percentage(number):
    return number * 100


def calculate_medicine(patient):
    # Because `predictions_object` is an "generator", weird stuff happens here
    predictions_object = classifier.predict(input_fn=lambda: input_fn(patient))
    for prediction_item in predictions_object:
        prediction_output = map(
            comma_to_percentage,
            prediction_item['probabilities']
        )
        prediction_output = list(prediction_output)

    # Plot graph
    title = 'Urinaltract infection treatment prediction'
    ind = np.arange(4)  # the x locations for the groups
    fig = plt.figure(
        figsize=(5, 7),
        num=title
    )
    ax = fig.add_subplot(111)

    result_bars = ax.bar(ind, prediction_output, 0.3, color='#777777')

    # Set labels on the x/y-axes
    ax.set_ylabel('Success Probability (%)')
    ax.set_xticks(ind)
    ax.set_xticklabels(
        (
            'Nitrofurantoine',
            'Amoxicilline',
            'Trimethoprim',
            'Ciprofloxacine'
        )
    )

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
    plt.title(title)
    plt.show()


def y_n_input(question):
    yes = {'yes', 'y', 'true', '1'}
    no = {'no', 'n', 'false', '0'}

    choice = input(question).lower()

    if choice in yes:
        return 1
    elif choice in no:
        return 0
    else:
        print("Please respond with 'yes' or 'no'")


def choose_medicine():
    # Get patient details
    leeftijd_cat = int(input(
        "Welke leeftijdscategorie (0 = 20-40, 1 = 40-60, 2 = 60-80, 3 = 80-100): "))
    geslacht = y_n_input("Is vrouw: ")
    urinewegklachten = y_n_input("Last van urinewegklachten: ")
    tekenen_van_weefselinvasie = y_n_input(
        "Heeft tekenen van weefselinvasie: ")
    diabetes = y_n_input("Heeft diabetes: ")
    zwanger = y_n_input("Is zwanger: ")
    patient_zit_in_nhg_risicogroep = y_n_input("Zit in NHG risico groep: ")

    patient = {
        'leeftijd_cat': [leeftijd_cat],
        'geslacht': [geslacht],
        'urinewegklachten': [urinewegklachten],
        'tekenen_van_weefselinvasie': [tekenen_van_weefselinvasie],
        'diabetes': [diabetes],
        'zwanger': [zwanger],
        'patient_zit_in_nhg_risicogroep': [patient_zit_in_nhg_risicogroep]
    }

    calculate_medicine(patient)

    choose_medicine()

choose_medicine()
