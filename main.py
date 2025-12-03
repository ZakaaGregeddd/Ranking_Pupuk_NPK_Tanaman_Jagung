import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# BAGIAN 1: LOAD & PREPROCESSING DATA
def load_and_clean_data(filename):
	# Baca CSV dengan delimiter titik koma
	df = pd.read_csv(filename, sep=';')

	# 1. Bersihkan Kolom Harga (Hapus 'Rp', '.', '/kg', dan spasi)
	# Mengubah "Rp 20.000/kg" menjadi angka 20000
	df['C1_Harga'] = df['harga'].astype(str).str.replace(r'[Rp. /kg]', '', regex=True).astype(float)

	# 2. Pastikan Kolom N, P, K adalah angka
	df['C2_N'] = pd.to_numeric(df['n'])
	df['C3_P'] = pd.to_numeric(df['p'])
	df['C4_K'] = pd.to_numeric(df['k'])

	# 3. Buat Kolom C5 (Bentuk) dari kolom 'jenis'
	# Logika: Jika mengandung kata 'cair', skor 5 (Cepat serap). Jika tidak, skor 3 (Padat).
	def get_bentuk_score(text):
		text = str(text).lower()
		if "cair" in text:
			return 5  # Cair
		elif "padat" in text:
			return 3  # Padat
		else:
			return 4  # Default/Granul/Mix, bisa disesuaikan
	df['C5_Bentuk'] = df['jenis'].apply(get_bentuk_score)

	return df


# Load Data Real Anda
df_alternatif = load_and_clean_data('data/data_pupuk.csv')


# Normalisasi min-max untuk kolom numerik
numerik_cols = ['C1_Harga', 'C2_N', 'C3_P', 'C4_K', 'C5_Bentuk']
df_normalized = df_alternatif.copy()
for col in numerik_cols:
	min_val = df_normalized[col].min()
	max_val = df_normalized[col].max()
	if max_val > min_val:
		df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
	else:
		df_normalized[col] = 0  # Jika semua nilai sama

# Export hasil numerikalisasi & normalisasi ke CSV di folder 'data'
preprocessed_csv_path = 'data/preprocessed_pupuk.csv'
df_normalized[['merk'] + numerik_cols].to_csv(preprocessed_csv_path, index=False)
print(f"Hasil preprocessing (numerik & normalisasi) diekspor ke {preprocessed_csv_path}")

# Pastikan folder results dan results/plots ada
os.makedirs('results/plots', exist_ok=True)

print("=== DATA HASIL PREPROCESSING ===")
print(df_alternatif[['merk', 'C1_Harga', 'C2_N', 'C3_P', 'C4_K', 'C5_Bentuk']].head())


# BAGIAN 2: METODE AHP (PEMBOBOTAN)

# Matriks Perbandingan Berpasangan (Sesuai Logika Fase Vegetatif)
# Urutan: [Harga, N, P, K, Bentuk]
# N (C2) dibuat dominan karena Fase Vegetatif.
ahp_matrix = np.array([
	# C1   C2   C3   C4   C5
	[1.0, 0.33, 2.0, 2.0, 1.0],  # C1: Harga (Lebih penting dr P & K, kalah sm N)
	[3.0, 1.0,  5.0, 5.0, 3.0],  # C2: Nitrogen (SANGAT PENTING/Dominan)
	[0.5, 0.2,  1.0, 1.0, 0.5],  # C3: Fosfor (Kurang penting di vegetatif)
	[0.5, 0.2,  1.0, 1.0, 0.5],  # C4: Kalium (Kurang penting di vegetatif)
	[1.0, 0.33, 2.0, 2.0, 1.0]   # C5: Bentuk (Setara harga)
])

# Hitung Bobot AHP
col_sum = ahp_matrix.sum(axis=0)
normalized_matrix = ahp_matrix / col_sum
weights = normalized_matrix.mean(axis=1)

# Label Kriteria
criteria_labels = ['C1: Harga', 'C2: Nitrogen', 'C3: Fosfor', 'C4: Kalium', 'C5: Bentuk']
weights_dict = dict(zip(['C1_Harga', 'C2_N', 'C3_P', 'C4_K', 'C5_Bentuk'], weights))

print("\n=== HASIL PEMBOBOTAN AHP (VEGETATIF) ===")
for k, v in zip(criteria_labels, weights):
	print(f"{k} : {v:.4f} ({v*100:.1f}%)")

# Cek Konsistensi (CR) - Optional tapi bagus untuk Skripsi
lambda_max = (np.dot(ahp_matrix, weights) / weights).mean()
CI = (lambda_max - 5) / (5 - 1)
RI = 1.12 # Konstanta untuk matriks 5x5
CR = CI / RI
print(f"Consistency Ratio (CR): {CR:.4f} (Valid jika < 0.1)")


# BAGIAN 3: METODE SAW (PERANGKINGAN)

# 1. Normalisasi Matriks
# C1 = Cost (Min/Nilai), C2-C5 = Benefit (Nilai/Max)
df_norm = df_alternatif.copy()

# Normalisasi Cost (Harga)
min_price = df_alternatif['C1_Harga'].min()
df_norm['norm_C1'] = min_price / df_alternatif['C1_Harga']

# Normalisasi Benefit (N, P, K, Bentuk)
for col in ['C2_N', 'C3_P', 'C4_K', 'C5_Bentuk']:
	max_val = df_alternatif[col].max()
	df_norm[f'norm_{col}'] = df_alternatif[col] / max_val

# 2. Hitung Skor Akhir (V)
df_norm['Skor_Akhir'] = (
	(df_norm['norm_C1'] * weights_dict['C1_Harga']) +
	(df_norm['norm_C2_N'] * weights_dict['C2_N']) +
	(df_norm['norm_C3_P'] * weights_dict['C3_P']) +
	(df_norm['norm_C4_K'] * weights_dict['C4_K']) +
	(df_norm['norm_C5_Bentuk'] * weights_dict['C5_Bentuk'])
)

# 3. Urutkan Ranking
df_final = df_norm.sort_values(by='Skor_Akhir', ascending=False).reset_index(drop=True)
df_final['Ranking'] = df_final.index + 1

print("\n=== HASIL AKHIR PERANGKINGAN SAW ===")
print(df_final[['Ranking', 'merk', 'jenis', 'C1_Harga', 'C2_N', 'Skor_Akhir']].head(10))

# Simpan hasil perankingan ke CSV di folder 'data'
ranking_csv_path = 'data/hasil_perankingan.csv'
df_final.to_csv(ranking_csv_path, index=False)
print(f"Hasil perankingan diekspor ke {ranking_csv_path}")

# Simpan hasil ke CSV (lama, bisa dihapus jika tidak perlu)
df_final.to_csv('results/hasil_rekomendasi_pupuk.csv', index=False)

# BAGIAN 4: VISUALISASI DATA (GRAFIK)
sns.set_style("whitegrid")

# Grafik 1: Bobot Kriteria AHP (Pie Chart)
plt.figure(figsize=(8, 6))
plt.pie(weights, labels=criteria_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Distribusi Bobot Kriteria (Prioritas Fase Vegetatif)', fontsize=14)
plt.tight_layout()
plt.savefig('results/plots/ahp_bobot_kriteria.png')
plt.show()

# Grafik 2: Top 10 Rekomendasi Pupuk (Bar Chart)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Skor_Akhir', y='merk', data=df_final.head(10), hue='merk', palette='viridis', legend=False)
plt.title('Top 10 Rekomendasi Pupuk Terbaik (Metode SAW)', fontsize=14)
plt.xlabel('Nilai Preferensi (V)')
plt.ylabel('Merk Pupuk')
plt.xlim(0, 1)
# Tambahkan angka pada setiap bar
for i, v in enumerate(df_final.head(10)['Skor_Akhir']):
	ax.text(v + 0.01, i, f"{v:.2f}", color='black', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/top10_rekomendasi_pupuk.png')
plt.show()


# Grafik 3: Analisis Sensitivitas (Harga vs Nitrogen)
# Membuktikan bahwa ranking tinggi punya N tinggi meskipun harga variatif
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
	data=df_final, 
	x='C1_Harga', 
	y='C2_N', 
	hue='Skor_Akhir', 
	size='Skor_Akhir',
	sizes=(50, 300), 
	palette='RdYlGn',
	edgecolor='black'
)
plt.title('Peta Sebaran Pupuk: Harga vs Kandungan N (Warna = Skor Akhir)', fontsize=14)
plt.xlabel('Harga (Rp/Kg)')
plt.ylabel('Kandungan Nitrogen (%)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Labeling beberapa top rank
for i in range(5):
	plt.text(
		df_final.C1_Harga[i]+500, 
		df_final.C2_N[i], 
		f"{df_final.Ranking[i]}. {df_final.merk[i]}", 
		fontsize=9, 
		fontweight='bold'
	)
plt.tight_layout()
plt.savefig('results/plots/sensitivitas_harga_nitrogen.png')
plt.show()


# Grafik 4a: Pergerakan Data - Nilai Asli Kriteria (tanpa harga)
plt.figure(figsize=(14, 8))
for col, color in zip(['C2_N', 'C3_P', 'C4_K', 'C5_Bentuk'], ['green', 'orange', 'red', 'purple']):
	plt.plot(df_alternatif['merk'], df_alternatif[col], label=f'Asli {col}', linestyle='-', color=color)
plt.xticks(rotation=90)
plt.legend()
plt.title('Pergerakan Data: Nilai Asli Kriteria (Tanpa Harga)')
plt.xlabel('Merk Pupuk')
plt.ylabel('Nilai Asli')
plt.tight_layout()
plt.savefig('results/plots/pergerakan_nilai_asli_kriteria.png')
plt.show()

# Grafik 4b: Pergerakan Data - Nilai Normalisasi Kriteria
plt.figure(figsize=(14, 8))
for col, color in zip(['C1_Harga', 'C2_N', 'C3_P', 'C4_K', 'C5_Bentuk'], ['blue', 'green', 'orange', 'red', 'purple']):
	norm_col = 'norm_C1' if col == 'C1_Harga' else f'norm_{col}'
	plt.plot(df_alternatif['merk'], df_norm[norm_col], label=f'Normalisasi {col}', color=color)
plt.xticks(rotation=90)
plt.legend()
plt.title('Pergerakan Data: Nilai Normalisasi Kriteria')
plt.xlabel('Merk Pupuk')
plt.ylabel('Nilai Normalisasi (0-1)')
plt.tight_layout()
plt.savefig('results/plots/pergerakan_normalisasi_kriteria.png')
plt.show()

# Grafik 5: Pergerakan Skor Akhir Seluruh Alternatif
plt.figure(figsize=(14, 6))
bars = plt.bar(df_final['merk'], df_final['Skor_Akhir'], color='teal')
plt.xticks(rotation=90)
plt.title('Pergerakan Skor Akhir Seluruh Alternatif (SPK)')
plt.xlabel('Merk Pupuk')
plt.ylabel('Skor Akhir')
# Tambahkan angka pada setiap bar
for bar in bars:
	height = bar.get_height()
	plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('results/plots/pergerakan_skor_akhir.png')
plt.show()
