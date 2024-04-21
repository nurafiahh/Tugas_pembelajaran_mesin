import pandas as pd
import numpy as np

# Membaca dataset dari file CSV
file_path = '/content/SCABIES SAPII.csv'  # Ganti dengan path file CSV Anda
data = pd.read_csv(file_path)

# Tampilkan dataset
print("Dataset:")
print(data)

# Hitung entropi
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value

entropi_total = entropy(data['hasil_tes_laboratorium'])
print(f"\nEntropi Total: {entropi_total}")

# Hitung gain untuk setiap atribut
def information_gain(data, attribute):
    values, counts = np.unique(data[attribute], return_counts=True)

    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = data[data[attribute] == value]
        subset_entropy = entropy(subset['hasil_tes_laboratorium'])
        weighted_entropy += count / len(data) * subset_entropy

    information_gain_value = entropi_total - weighted_entropy
    return information_gain_value

attributes = ['jenis_kelamin', 'usia', 'gatal', 'kerontokan_bulu', 'kerak_pada_kulit']
for attribute in attributes:
    gain = information_gain(data, attribute)
    print(f"Gain for {attribute}: {gain}")

    # DATA BERBENTUK STRING AKAN DIJADIKAN NUMERIK
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
data['jenis_kelamin'] = enc.fit_transform(data['jenis_kelamin'].values)
data['gatal'] = enc.fit_transform(data['gatal'].values)
data['kerontokan_bulu'] = enc.fit_transform(data['kerontokan_bulu'].values)
data['kerak_pada_kulit'] = enc.fit_transform(data['kerak_pada_kulit'].values)
data['hasil_tes_laboratorium'] = enc.fit_transform(data['hasil_tes_laboratorium'].values)
data.head(600)

# PEMISAHAN DATA VARIABEL DAN DATA TARGET
atr_data = data.drop(columns = 'hasil_tes_laboratorium')
atr_data.head(500)

cls_data = data['hasil_tes_laboratorium']
cls_data.head(500)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.stats import entropy

# MEMISAHKAN DATA TRAIN DAN DATA TES
xtrain, xtest, ytrain, ytest = train_test_split(atr_data, cls_data, test_size=0.2, random_state=500)
tree_data = DecisionTreeClassifier(random_state=500)
tree_data.fit(xtrain, ytrain)

# MEMBUAT VARIABEL UNTUK MENGITUNG CONFUSION MATRIK DAN AKURASI
model = DecisionTreeClassifier(criterion='entropy', random_state=500)
model.fit(xtrain, ytrain)

# MEMBUAT VARIABEL UNTUK MENGHITUNG GAIN INFORMATION
initial_entropy = entropy(ytrain.value_counts(normalize=True), base=2)

# MEMBUAT PREDIKSI
ypred = model.predict(xtest)

# MENGHITUNG CONFUSIION MATRIKS
cm = confusion_matrix(ytest, ypred)
print("Confusion Matrix:")
print(cm)

# MENGHITUNG KALKUASI SELURUH MATRIKS
accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)

# MENAMPILKAN EVALUASI MATRIKS
print("\nEvaluasi Model:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# MENGIMPOR LIBRARY UNTUK MEMBUAT GAMBAR POHON KEPUTUSANYYA
from sklearn.tree import export_graphviz
export_graphviz(tree_data, out_file="tree_hasil_tes_laboratorium", class_names=["POSITIF","NEGATIF"],
                feature_names=atr_data.columns, impurity=False, filled=True)

# MENGIMPORT LIBRARY
import graphviz

# Membaca file DOT
with open("tree_hasil_tes_laboratorium") as fig:
    dot_graph = fig.read()

# Membuat dan menampilkan objek graph dari DOT
graph = graphviz.Source(dot_graph)
graph.render("tree_hasil_tes_laboratorium", format='png', cleanup=True)
graph.view("tree_hasil_tes_laboratorium")

# Membuat model Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)

# Mencetak aturan-aturan hasil klasifikasi
tree_rules = export_text(clf, feature_names=atr_data.columns.tolist())
print(tree_rules)