# hada likhdama


import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
sc = StandardScaler()
rf = RandomForestRegressor()

def predict(x_train, y_train, x_test, y_test):
    print("Dimensions of x_train:", x_train.shape)
    print("Dimensions of y_train:", y_train.shape)
    print("Dimensions of x_test:", x_test.shape)
    print("Dimensions of y_test:", y_test.shape)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).reshape(-1, 1)
    mse = mean_squared_error(y_test, y_pred)
    acc = round(rf.score(x_train, y_train), 2) * 100
    return mse, acc, y_pred

def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            return pd.read_csv(file_path, sep='\s+', header=None)
        else:
            messagebox.showerror("Error", "Unsupported file format.\
                                 Please select a CSV or TXT file.")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")
        return None

def browse_files(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(tk.END, file_path)
def download_pred_button_click(y_pred_df):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        y_pred_df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", "Predicted values saved successfully.")
def predict_button_click(train_entry, test_entry, y_test_entry, root):
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
    messagebox.showinfo("Success", "Files successfully loaded.")
    y_train = train_df.iloc[:, -1:].values.ravel()
    X_train = sc.fit_transform(train_df.iloc[:, :-1])
    test_df.columns = train_df.columns
    X_test = sc.transform(test_df.iloc[:, :-1])
    y_test = y_test_df.iloc[:, -1:].values.ravel()
    num_repeats = int(np.ceil(X_test.shape[0] / y_test.shape[0]))
    y_test = np.repeat(y_test, num_repeats)[:X_test.shape[0]]
    mse, acc, y_pred= predict(X_train, y_train, X_test, y_test)
    # Créer un DataFrame pandas avec les résultats prédits
    y_pred_df = pd.DataFrame(y_pred, columns=["Predicted RUL"])
    print(f'Mean Squared Error: {mse}')
    print(f'Accuracy: {acc}')
    print("Train Data:")
    print(train_df.head())
    print("\nTest Data:")
    print(test_df.head())
    print("\ny_test Data:")
    print(y_test_df.head())
    # Affichage des résultats
    result_window = tk.Toplevel(root)
    result_window.title("Prediction Results")
    result_window.geometry("800x400")
    # Créer un widget Text pour afficher les résultats
    result_text = Text(result_window, wrap="word", width=100, height=20)
    result_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    # Insérer les résultats dans le widget Text
    result_text.tag_configure("bold", font=("Helvetica",9, "bold"))  # Configuration du tag pour le texte en gras
    result_text.insert(END, "Loss (MSE): ", "bold")
    result_text.insert(END, f'{mse}\n')
    result_text.insert(END, "Accuracy: ", "bold")
    result_text.insert(END, f'{acc}\n\n')
    result_text.insert(END, "Predicted Values:\n", "bold")
    result_text.insert(END,y_pred)
    # Créer un bouton pour télécharger le DataFrame y_pred
    download_button = tk.Button(result_window, text="Download Predicted Values",\
    command=lambda: download_pred_button_click(y_pred_df))
    download_button.grid(row=2, column=0, pady=10)  
    download_button.configure(bg='blue', fg='white',height=2, width=30)   
    root.result_window = result_window
def main():
    root = tk.Tk()
    root.title("Predictive Maintenance using NASA turbofan")
    # Titre de la page
    title_label = tk.Label(root, text="Predictive Maintenance System.\
        using NASA turbofan", font=("Helvetica", 16))
    title_label.grid(row=0, column=2, columnspan=3, pady=20)
    root.geometry("800x400")
    # Définir les dimensions et la position de la fenêtre
    train_label = tk.Label(root, text="Train Data:")
    train_label.grid(row=1, column=1)
    train_entry = tk.Entry(root, width=100)
    train_entry.grid(row=1, column=2)
    train_button = tk.Button(root, text="Browse", \
        command=lambda: browse_files(train_entry))
    train_button.grid(row=1, column=3)
    train_button.configure(bg='red',height=1,width=15,fg='white') 
    test_label = tk.Label(root, text="Test Data:")
    test_label.grid(row=2, column=1)
    test_entry = tk.Entry(root, width=100)
    test_entry.grid(row=2, column=2)
    test_button = tk.Button(root, text="Browse" \
        , command=lambda: browse_files(test_entry))
    test_button.grid(row=2, column=3)
    test_button.configure(bg='red',height=1,width=15,fg='white')
    y_test_label = tk.Label(root, text="y_test Data:")
    y_test_label.grid(row=3, column=1)
    y_test_entry = tk.Entry(root, width=100)
    y_test_entry.grid(row=3, column=2)
    y_test_button = tk.Button(root, text="Browse" \
        , command=lambda: browse_files(y_test_entry))
    y_test_button.grid(row=3, column=3)
    y_test_button.configure(bg='red',height=1,width=15,fg='white')
    predict_button = tk.Button(root, text="Predict",command=lambda:\
    predict_button_click(train_entry, test_entry, y_test_entry, root))
    predict_button.grid(row=4, column=2)
    predict_button.configure(bg='blue',height=1,width=15,fg='white') 
    root.mainloop()
if __name__ == "__main__":
    main()
