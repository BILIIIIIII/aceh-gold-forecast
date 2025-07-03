import os
import sys

# Ganti 'api.main' dengan lokasi file utama FastAPI kamu
# Formatnya adalah 'nama_folder.nama_file'
from api.main import app as application

# Menambahkan direktori proyek ke path Python
sys.path.insert(0, os.path.dirname(__file__))