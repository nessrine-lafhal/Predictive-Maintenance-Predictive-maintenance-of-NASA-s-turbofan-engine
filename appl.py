import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

sc = StandardScaler()
rf = RandomForestRegressor()

def predict(x_train, y_train, x_test, y_test):
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    acc = round(rf.score(x_train, y_train), 2) * 100
    
    plt.plot(y_test, color="blue", linewidth=2.5, linestyle="-", label="Actual")
    plt.plot(y_pred, color='red', linewidth=2.5, linestyle="-", label="Predicted")
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.show()
    
    return mse, acc

def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            return pd.read_csv(file_path, sep='\s+', header=None)
        else:
            messagebox.showerror("Error", "Unsupported file format. Please select a CSV or TXT file.")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")
        return None

def browse_files(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(tk.END, file_path)

def predict_button_click(train_entry, test_entry, y_test_entry):
    train_file = train_entry.get()
    test_file = test_entry.get()
    y_test_file = y_test_entry.get()
    if not (train_file and test_file and y_test_file):
        messagebox.showerror("Error", "Please select all files.")
        return

    train_df = load_data(train_file)
    test_df = load_data(test_file)
    y_test_df = load_data(y_test_file)

    if train_df is None or test_df is None or y_test_df is None:
        messagebox.showerror("Error", "Failed to load data files.")
        return

    train_df.columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 
                        's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
                        's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 
                        's_20', 's_21']

    test_df.columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 
                       's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
                       's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 
                       's_20', 's_21']

    y_test_df.columns = ['RUL']

    train_data = train_df.copy()
    test_data = test_df.copy()
    y_test_data = y_test_df.copy()

    train_data = train_data.drop(['setting_3', 's_1', 's_10', 's_18', 's_19'], axis=1)
    test_data = test_data.drop(['setting_3', 's_1', 's_10', 's_18', 's_19'], axis=1)

    train_data['RUL'] = train_data.groupby('unit_nr')['time_cycles'].transform(max) - train_data['time_cycles']
    test_data['RUL'] = y_test_data['RUL'].repeat(test_data.groupby('unit_nr').size()).values

    final_train_data = train_data[['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_13', 's_15', 's_17', 's_20', 's_21']]
    final_test_data = test_data[['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_13', 's_15', 's_17', 's_20', 's_21']]

    X_train = final_train_data.drop('RUL', axis=1)
    y_train = final_train_data['RUL']

    X_test = final_test_data.drop('RUL', axis=1)
    y_test = final_test_data['RUL']

    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    mse, acc = predict(X_train_scaled, y_train, X_test_scaled, y_test)
    
    messagebox.showinfo("Success", f"Mean Squared Error: {mse}, Accuracy: {acc}")

def main():
    root = tk.Tk()
    root.title("Predictive Maintenance using NASA turbofan")
    root.geometry("600x400")

    train_label = tk.Label(root, text="Train Data:")
    train_label.grid(row=0, column=0)
    train_entry = tk.Entry(root, width=50)
    train_entry.grid(row=0, column=1)
    train_button = tk.Button(root, text="Browse", command=lambda: browse_files(train_entry))
    train_button.grid(row=0, column=2)

    test_label = tk.Label(root, text="Test Data:")
    test_label.grid(row=1, column=0)
    test_entry = tk.Entry(root, width=50)
    test_entry.grid(row=1, column=1)
    test_button = tk.Button(root, text="Browse", command=lambda: browse_files(test_entry))
    test_button.grid(row=1, column=2)

    y_test_label = tk.Label(root, text="y_test Data:")
    y_test_label.grid(row=2, column=0)
    y_test_entry = tk.Entry(root, width=50)
    y_test_entry.grid(row=2, column=1)
    y_test_button = tk.Button(root, text="Browse", command=lambda: browse_files(y_test_entry))
    y_test_button.grid(row=2, column=2)

    predict_button = tk.Button(root, text="Predict", command=lambda: predict_button_click(train_entry, test_entry, y_test_entry))
    predict_button.grid(row=3, column=1)

    root.mainloop()

if __name__ == "__main__":
    main()
