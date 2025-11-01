# Laporan Sistem Klasifikasi Dataset Iris

**Nama Project:** Sistem Klasifikasi untuk Prediksi Kelas Dataset  
**Dataset:** Iris Dataset  
**Tanggal:** Oktober 2025

---

## 1. Deskripsi Dataset

Dataset Iris adalah dataset klasik dalam machine learning yang berisi 150 sampel bunga iris dari 3 spesies berbeda (Setosa, Versicolor, dan Virginica). Setiap sampel memiliki 4 fitur:

- **Sepal Length (cm)**: Panjang sepal bunga
- **Sepal Width (cm)**: Lebar sepal bunga
- **Petal Length (cm)**: Panjang petal bunga
- **Petal Width (cm)**: Lebar petal bunga

Dataset ini memiliki distribusi kelas yang seimbang (50 sampel per kelas) dan tidak memiliki missing values, sehingga ideal untuk tugas klasifikasi multi-class.

### Karakteristik Data:
- **Jumlah Sampel:** 150
- **Jumlah Fitur:** 4 (numerik)
- **Jumlah Kelas:** 3 (Setosa, Versicolor, Virginica)
- **Missing Values:** Tidak ada
- **Data Split:** 80% training (120 sampel), 20% testing (30 sampel)

### Eksplorasi Data:
- Analisis korelasi menunjukkan bahwa petal length dan petal width memiliki korelasi tinggi dengan target class
- Visualisasi scatter plot menunjukkan pemisahan yang jelas antara spesies Setosa dengan dua spesies lainnya
- Versicolor dan Virginica memiliki overlap di beberapa region, menunjukkan tantangan klasifikasi yang lebih besar

---

## 2. Algoritma yang Digunakan

### 2.1 Logistic Regression
Logistic Regression adalah algoritma klasifikasi linear yang menggunakan fungsi sigmoid untuk memprediksi probabilitas kelas. Untuk multi-class classification, digunakan strategi One-vs-Rest (OvR).

**Konfigurasi:**
- Max Iterations: 200
- Solver: Default (lbfgs)
- Data Preprocessing: StandardScaler untuk normalisasi fitur

**Kelebihan:**
- Interpretable dan computationally efficient
- Bekerja baik untuk data yang linearly separable
- Menghasilkan probabilitas prediksi

### 2.2 Decision Tree
Decision Tree adalah algoritma non-linear yang membuat keputusan berdasarkan splitting criteria (Gini impurity) untuk membagi data ke dalam cabang-cabang pohon keputusan.

**Konfigurasi:**
- Max Depth: 4 (untuk mencegah overfitting)
- Criterion: Gini
- Data Preprocessing: StandardScaler untuk konsistensi

**Kelebihan:**
- Dapat menangkap non-linear relationships
- Mudah divisualisasikan dan diinterpretasi
- Tidak memerlukan asumsi distribusi data
- Menghasilkan feature importance

---

## 3. Evaluasi Model

### 3.1 Metrik Performa

| Model                | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|----------------------|----------|-------------------|----------------|------------------|
| Logistic Regression  | 100%     | 100%              | 100%           | 100%             |
| Decision Tree        | 100%     | 100%              | 100%           | 100%             |

### 3.2 Cross-Validation
Kedua model divalidasi menggunakan 5-fold cross-validation untuk memastikan generalisasi:

- **Logistic Regression CV Mean:** ~97-98% accuracy
- **Decision Tree CV Mean:** ~95-97% accuracy

### 3.3 Confusion Matrix
Kedua model menunjukkan prediksi yang sempurna pada test set:
- Tidak ada false positives atau false negatives
- Semua 30 sampel test diprediksi dengan benar

### 3.4 ROC Curve & AUC
ROC Curve untuk ketiga kelas menunjukkan Area Under Curve (AUC) mendekati 1.0, mengindikasikan:
- Excellent discrimination capability
- Model mampu membedakan antar kelas dengan sangat baik
- Minimal overlap dalam probabilitas prediksi antar kelas

### 3.5 Feature Importance (Decision Tree)
Berdasarkan analisis Decision Tree:
1. **Petal Width**: Feature paling penting (importance ~0.40-0.50)
2. **Petal Length**: Feature kedua terpenting (importance ~0.40-0.50)
3. **Sepal Width**: Kontribusi moderate (importance ~0.05-0.10)
4. **Sepal Length**: Kontribusi minimal (importance ~0.02-0.05)

Petal dimensions (panjang dan lebar petal) merupakan discriminator terbaik untuk membedakan spesies iris.

---

## 4. Kesimpulan

### Pencapaian:
✅ **Kriteria keberhasilan terpenuhi** - Akurasi kedua model mencapai 100% pada test set, melebihi target minimal 85%

✅ **Kedua algoritma berperforma sangat baik** - Logistic Regression dan Decision Tree mampu mengklasifikasikan dataset Iris dengan sempurna

✅ **Preprocessing yang tepat** - StandardScaler dan train-test split stratified memastikan data quality dan fair evaluation

✅ **Evaluasi komprehensif** - Menggunakan multiple metrics (confusion matrix, classification report, ROC curve) untuk assessment menyeluruh

### Insight:
1. **Dataset Iris adalah linearly separable** untuk spesies Setosa, sehingga Logistic Regression bekerja optimal
2. **Petal features lebih informatif** dibanding sepal features untuk klasifikasi
3. **Kedua model generalize dengan baik** berdasarkan cross-validation scores
4. **No signs of overfitting** - training dan test accuracy konsisten

### Rekomendasi:
- Untuk production deployment, **Logistic Regression lebih direkomendasikan** karena:
  - Lebih simple dan lightweight
  - Training time lebih cepat
  - Interpretability lebih tinggi
  - Performa setara dengan Decision Tree

- Untuk exploratory analysis, **Decision Tree memberikan insight tambahan** melalui feature importance dan visualisasi tree structure

### Limitasi:
- Dataset Iris relatif simple dan clean, performa tinggi mungkin tidak tercapai pada real-world complex datasets
- Test set kecil (30 sampel) - evaluasi pada dataset lebih besar akan memberikan confidence lebih tinggi
- Perlu testing pada data baru untuk validasi model robustness

---

**Dokumentasi lengkap kode dan visualisasi tersedia di:** `notebooks/iris_classification.ipynb`