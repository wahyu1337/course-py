import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import os
import pandas as pd

# Fungsi untuk membaca dan mengkonversi gambar ke skala gret
def konver_skala_grey(image_path):
  image = cv2.imread(image_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return gray_image

# Fungsi untuk menghitung GLCM dan ekstraksi fitur tekstur 
def ekstrak_fitur_glcm(gray_image, distances, angles):
  glcm = graycomatrix(gray_image, distances=distances, angles=angles,levels=256, symmetric=True, normed=True)
  contrast = graycoprops(glcm, 'contrast')
  dissimilarity = graycoprops(glcm, 'dissimilarity')
  homogeneity = graycoprops(glcm, 'homogeneity')
  energy = graycoprops(glcm, 'energy')
  correlation = graycoprops(glcm, 'correlation')
  return contrast, dissimilarity, homogeneity, energy, correlation

# Fungsi untuk menampilkan dan mencetak hasil ekstraksi fitur dari beberapa gambar
def proses_glcm(image_paths, label):
  distances = [1, 2] # Jarak antar piksel
  angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # sudut pixel yang digunakan
  results = [] # list untuk menyimpan hasil
  n = 0
  for image_path in image_paths:
    print('Memproses gambar '+image_path)
    #memanggil fungsi konversi gambar ke skala grey
    gray_image = konver_skala_grey(image_path)
    #memanggil fungsi ekstraksi fitur
    contrast, dissimilarity, homogeneity, energy, correlation = ekstrak_fitur_glcm(gray_image, distances, angles)
    # Simpan hasil ke dalam list untuk disimpan ke file csv
    results.append({
      'Image': os.path.basename(image_path), 'Contrast': np.mean(contrast), 'Dissimilarity': np.mean(dissimilarity), 'Homogeneity': np.mean(homogeneity), 'Energy': np.mean(energy), 'Correlation': np.mean(correlation), 'Status': label
    })
    n = n*1
  # Simpan hasil ke fil CSV
  print('Menyimpan hasil pada hasil_ekstraksi_fitur.csv')
  results_df = pd.DataFrame(results)
  results_df.to_csv('hasil_ekstraksi_tekstur.csv', index=False)

if __name__ == "__main__":
  image_paths = [os.path.join('rusak', f) for f in os.listdir('rusak') if f.endswith('.jpg')] + [os.path.join('normal', f) for f in os.listdir('normal')if f.endswith('.jpg')]

  image_label = [os.path.dirname(path) for path in image_paths]
  proses_glcm(image_paths, image_label)
