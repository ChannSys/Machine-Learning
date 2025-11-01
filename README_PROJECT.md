# Sistem Klasifikasi Dataset Iris

Project Machine Learning untuk memprediksi spesies bunga Iris menggunakan dua algoritma klasifikasi: **Logistic Regression** dan **Decision Tree**.

---

## 📁 Struktur Project

```
Machine Learning/
├── readme.md                      # Product Requirements Document (PRD)
├── requirements.txt               # Dependencies Python
├── README_PROJECT.md             # Panduan penggunaan project (file ini)
├── data/
│   └── iris.csv                  # Dataset Iris
├── notebooks/
│   └── iris_classification.ipynb # Jupyter Notebook lengkap
├── scripts/
│   └── train_model.py            # Python script untuk training
└── reports/
    └── laporan_klasifikasi.md    # Laporan hasil evaluasi (3 halaman)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cara Menjalankan

#### Opsi A: Menggunakan Jupyter Notebook (Rekomendasi)

```bash
jupyter notebook notebooks/iris_classification.ipynb
```

Jupyter Notebook berisi:
- Exploratory Data Analysis (EDA) dengan visualisasi
- Data preprocessing
- Model training (Logistic Regression & Decision Tree)
- Evaluasi lengkap dengan confusion matrix, ROC curve, dan feature importance
- Kesimpulan

#### Opsi B: Menggunakan Python Script

```bash
python scripts/train_model.py
```

Script akan menampilkan:
- Dataset info dan class distribution
- Training accuracy dan test accuracy kedua model
- Cross-validation scores
- Classification report lengkap
- Feature importance
- Kesimpulan

---

## 📊 Hasil Evaluasi

### Performa Model

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 93.33%   | 93.33%    | 93.33% | 93.33%   |
| Decision Tree        | 93.33%   | 93.33%    | 93.33% | 93.33%   |

✅ **Kriteria keberhasilan terpenuhi** - Akurasi kedua model di atas 85%

### Feature Importance (Decision Tree)

1. **Petal Length** - 56.56% importance
2. **Petal Width** - 41.12% importance
3. **Sepal Width** - 1.69% importance
4. **Sepal Length** - 0.63% importance

**Insight:** Petal dimensions (panjang dan lebar petal) adalah fitur paling penting untuk klasifikasi spesies Iris.

---

## 📖 Dokumentasi Lengkap

### Product Requirements Document
Lihat `readme.md` untuk detail tujuan, lingkup pekerjaan, dan kriteria keberhasilan project.

### Laporan Evaluasi
Lihat `reports/laporan_klasifikasi.md` untuk laporan lengkap (maksimal 3 halaman) yang mencakup:
- Deskripsi dataset
- Algoritma yang digunakan
- Evaluasi model
- Kesimpulan dan rekomendasi

---

## 🛠️ Tech Stack

- **Python 3.13**
- **NumPy** - Komputasi numerik
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualisasi data
- **Scikit-learn** - Machine learning library
- **Jupyter Notebook** - Interactive development

---

## 📝 Metodologi

### 1. Exploratory Data Analysis (EDA)
- Analisis distribusi kelas
- Correlation matrix
- Visualisasi scatter plots
- Pengecekan missing values

### 2. Data Preprocessing
- Train-test split (80%-20%)
- Stratified sampling untuk balanced classes
- StandardScaler untuk normalisasi fitur

### 3. Model Training
- **Logistic Regression** (max_iter=200, random_state=42)
- **Decision Tree** (max_depth=4, random_state=42)
- 5-fold Cross-Validation

### 4. Evaluasi
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve & AUC
- Feature Importance

---

## 🎯 Kesimpulan

1. ✅ Kedua model mencapai akurasi **93.33%**, melebihi target minimal **85%**
2. ✅ Tidak ada overfitting - training dan test accuracy konsisten
3. ✅ Petal features (length & width) adalah fitur terpenting untuk klasifikasi
4. ✅ Model telah memenuhi semua kriteria keberhasilan dalam PRD

### Rekomendasi
- **Logistic Regression** lebih direkomendasikan untuk production deployment karena lebih simple, lightweight, dan performa setara
- **Decision Tree** cocok untuk exploratory analysis karena mudah diinterpretasi dan memberikan feature importance

---

## 👨‍💻 Author

Project ini dibuat sebagai tugas Machine Learning Semester 5.

---

## 📅 Timeline Pengerjaan

| Hari   | Aktivitas                                              | Status |
| ------ | ------------------------------------------------------ | ------ |
| Hari 1 | Pemilihan dataset, EDA, preprocessing data             | ✅      |
| Hari 2 | Implementasi model Logistic Regression & Decision Tree | ✅      |
| Hari 3 | Evaluasi model dan visualisasi hasil                   | ✅      |
| Hari 4 | Penyusunan laporan dan finalisasi project              | ✅      |

---

**Project Status:** ✅ COMPLETED - Semua deliverables terpenuhi sesuai PRD
