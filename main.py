import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as pdr

from tkinter import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM


def myClick():

    # Load Data
    company = tickerEntry.get()
    epochsAmount = epochsEntry.get()
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)
    data = pdr.DataReader(company, 'yahoo', start, end)

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build The Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=int(epochsAmount), batch_size=32)

    """TEST THE MODEL ACCURACY ON EXISTING DATA"""

    # Load Test Data
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = pdr.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    # Double check here for the last one which is value
    model_inputs = total_dataset[len(
        total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Predict Next Day
    real_data = [
        model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(
        real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    predictionLabel = Label(
        root, text=f"Estimated Closing Price Tommorow: {prediction}")
    predictionLabel.grid(row=4, columnspan=2)


# GUI setup
root = Tk()
root.title("AI Python Stock Price Predictor GUI")

# Labels, Entries, and Buttons
tickerLabel = Label(root, text="Ticker Symbol: ")
epochsLabel = Label(root, text="# of Epochs: ")
noteLabel = Label(root, text="Note: More epochs takes more time :)")

tickerEntry = Entry(root, width=20)
tickerEntry.insert(0, "AAPL")
epochsEntry = Entry(root, width=20)
epochsEntry.insert(0, 3)

myButton = Button(root, text="Calculate", padx=50, command=myClick)

tickerLabel.grid(row=0, column=0)
tickerEntry.grid(row=0, column=1)
epochsLabel.grid(row=1, column=0)
epochsEntry.grid(row=1, column=1)
noteLabel.grid(row=2, columnspan=2)
myButton.grid(row=3, columnspan=2)

root.mainloop()
