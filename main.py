from mainwindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication
import sys

from PyQt5.QtWidgets import QMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.hikayeSecButton.clicked.connect(ui.hikaye_sec)
    ui.analizEtButton.clicked.connect(ui.analiz_et)
 
    sys.exit(app.exec_())
    