import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QTextEdit
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

class TomographThread(QThread):
    status_update = pyqtSignal(str)  # Sinyal untuk memperbarui status

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        keliling = self.data.get("keliling")
        jumlah_sensor = self.data.get("jumlah_sensor")
        data_pengukuran = np.array(self.data.get("data_pengukuran"))

        radius = keliling / (2 * np.pi)
        resolution = 300
        num_points = jumlah_sensor

        self.status_update.emit("Menghitung radius...")
        x = np.linspace(-radius, radius, resolution)
        y = np.linspace(-radius, radius, resolution)
        X, Y = np.meshgrid(x, y)

        distance_from_center = np.sqrt(X**2 + Y**2)
        mask = distance_from_center <= radius

        self.status_update.emit("Menginisialisasi data...")
        data = np.zeros((resolution, resolution))
        overlap_count = np.zeros((resolution, resolution))

        self.status_update.emit("Menentukan titik sensor...")
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]

        weights = data_pengukuran
        weights_t = weights.transpose()
        for i in range(8):
            for j in range(8):
                weights[i][j] = weights[i][j] + weights_t[i][j]

        compensate = np.array([
            [1, 65.60287412, 121.21830535, 158.37934843, 171.42857143, 158.37934843, 121.21830535, 65.60287412],
            [65.60287412, 1, 65.60287412, 121.21830535, 158.37934843, 171.42857143, 158.37934843, 121.21830535],
            [121.21830535, 65.60287412, 1, 65.60287412, 121.21830535, 158.37934843, 171.42857143, 158.37934843],
            [158.37934843, 121.21830535, 65.60287412, 1, 65.60287412, 121.21830535, 158.37934843, 171.42857143],
            [171.42857143, 158.37934843, 121.21830535, 65.60287412, 1, 65.60287412, 121.21830535, 158.37934843],
            [158.37934843, 171.42857143, 158.37934843, 121.21830535, 65.60287412, 1, 65.60287412, 121.21830535],
            [121.21830535, 158.37934843, 171.42857143, 158.37934843, 121.21830535, 65.60287412, 1, 65.60287412],
            [65.60287412, 121.21830535, 158.37934843, 171.42857143, 158.37934843, 121.21830535, 65.60287412, 1]
        ])

        weights = weights / compensate * 171.42857143

        self.status_update.emit("Mengisi bobot dalam elips...")
        total_iterations = num_points * num_points  # Total iterasi
        for i in range(num_points):
            for j in range(num_points):
                x1, y1 = points[i]
                x2, y2 = points[j]
                weight = weights[i, j]

                a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if (abs(j-i) < (num_points)/2):
                    c = 1 - (abs(j-i) / (num_points/2))
                elif(abs(j-i) > (num_points)/2):
                    c = 1 - (num_points - abs(j-i)) / (num_points/2)
                else:
                    c = 0.1

                b = c * a

                for xi in range(resolution):
                    for yi in range(resolution):
                        dx = X[xi, yi] - (x1 + x2) / 2
                        dy = Y[xi, yi] - (y1 + y2) / 2

                        theta = np.arctan2(y2 - y1, x2 - x1)
                        dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
                        dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)

                        if ((a != 0) & (b != 0)):
                            if (dx_rot**2 / a**2 + dy_rot**2 / b**2) <= 1:
                                data[xi, yi] += weight
                                overlap_count[xi, yi] += 1

                # Update status dengan iterasi saat ini
                current_iteration = i * num_points + j + 1  # Hitung iterasi saat ini
                self.status_update.emit(f"Mengisi bobot: Iterasi {current_iteration} dari {total_iterations}")

        self.status_update.emit("Terapkan masking...")
        data = np.where(mask, data, np.nan)
        overlap_count = np.where(mask, overlap_count, np.nan)

        self.status_update.emit("Penyesuaian bobot...")
        adjusted_data = np.where(overlap_count > 0, data / overlap_count, np.nan)

        self.status_update.emit("Terapkan Gaussian blur...")
        blurred_data = gaussian_filter(adjusted_data, sigma=2)

        self.status_update.emit("Menyiapkan visualisasi...")
        plt.imshow(blurred_data, extent=(-radius, radius, -radius, radius), origin='lower', cmap='RdYlGn')
        plt.colorbar(label='Cepat Rambat (m/s)')

        plt.grid(True, linestyle='--', color='blue', alpha=0.2)

        for idx, (px, py) in enumerate(points):
            plt.scatter(px, py, color='white')
            plt.text(px, py, f's{idx + 1}', color='black', fontsize=12, ha='center', va='center', weight='bold')

        plt.title('Visualisasi Batang')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.savefig('E:\\Windows User Data\\Desktop\HALO\Tesis\\Codebase\\Hasil Visualisasi 17 Maret.png', format='png')
        plt.show()

        self.status_update.emit("Visualisasi selesai!")

class JSONLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Terawang - Visualisasi')
        self.setGeometry(100, 100, 400, 800)
        self.setWindowIcon(QIcon("icon.png"))

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Tambahkan label untuk ikon dan judul
        headerLayout = QHBoxLayout()
        
        # Gambar Judul
        iconLabel = QLabel(self)
        iconLabel.setGeometry(0, 0, 25, 25)
        iconPixmap = QPixmap('icon.png')
        iconLabel.setPixmap(iconPixmap)
        iconLabel.setScaledContents(True)
        
        titleLabel = QLabel("Terawang - Visualisasi", self)
        titleLabel.setFont(QFont("Arial", 64))
        titleLabel.setStyleSheet("font-weight: bold; color: #472d23;")
        
        headerLayout.addWidget(iconLabel)
        headerLayout.addWidget(titleLabel)

        self.layout.addLayout(headerLayout)

        self.loadButton = QPushButton('Load File Hasil Pengukuran (JSON)', self)
        self.loadButton.setStyleSheet("font-size: 30px;")
        self.loadButton.clicked.connect(self.loadJSON)
        self.layout.addWidget(self.loadButton)

        self.resultText = QTextEdit(self)
        self.resultText.setReadOnly(True)
        self.resultText.setStyleSheet("font-size: 20px;")
        self.layout.addWidget(self.resultText)

        self.statusLabel = QLabel("", self)
        self.statusLabel.setFont(QFont("Arial", 20))
        self.layout.addWidget(self.statusLabel)

        self.createTomographButton = None

        self.setLayout(self.layout)

        self.data = {}

    def loadJSON(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            with open(fileName, 'r') as file:
                self.data = json.load(file)
                self.processData(self.data)
                self.addCreateTomographButton()

    def processData(self, data):
        species = data.get("species")
        keliling = data.get("keliling")
        jumlah_sensor = data.get("jumlah_sensor")
        ketinggian = data.get("ketinggian")
        data_pengukuran = data.get("data_pengukuran")
        
        display_text = (
            f"Spesies: {species}\n"
            f"Keliling: {keliling}\n"
            f"Jumlah Sensor: {jumlah_sensor}\n"
            f"Ketinggian: {ketinggian}\n"
            f"Data Pengukuran:\n"
        )
        
        for row in data_pengukuran:
            display_text += f"{row}\n"
        
        self.resultText.setText(display_text)

    def addCreateTomographButton(self):
        if not self.createTomographButton:
            self.createTomographButton = QPushButton('Buat Tomograf', self)
            self.createTomographButton.setStyleSheet("font-size: 30px;")
            self.createTomographButton.clicked.connect(self.createTomograph)
            self.layout.addWidget(self.createTomographButton)

    def createTomograph(self):
        self.statusLabel.setText("Memulai proses...")
        self.thread = TomographThread(self.data)
        self.thread.status_update.connect(self.statusLabel.setText)  # Menghubungkan sinyal
        self.thread.start()  # Menjalankan thread

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JSONLoaderApp()
    ex.show()
    sys.exit(app.exec_())
