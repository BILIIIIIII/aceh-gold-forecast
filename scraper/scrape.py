import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import csv
import time
from tqdm import tqdm

# Mapping bulan (angka) -> string nama bulan (bahasa Indonesia).
# Sesuaikan jika situs pakai bahasa Inggris, mis. "January", "February", dll.
MONTH_NAMES = {
    1:  "Januari",
    2:  "Februari",
    3:  "Maret",
    4:  "April",
    5:  "Mei",
    6:  "Juni",
    7:  "Juli",
    8:  "Agustus",
    9:  "September",
    10: "Oktober",
    11: "November",
    12: "Desember"
}

def scrape_gold_price(year: int, month_str: str, day: int, debug: bool = False):
    """
    Scrape harga emas (USD, IDR) pada tanggal tertentu.
    Return (usd_value, idr_value) atau (None, None) jika data tidak ditemukan.

    Jika debug=True, maka akan mencetak info detail seperti URL, data yang didapat.
    """
    # Contoh URL: "https://harga-emas.org/history-harga/2013/September/1/"
    url = f"https://harga-emas.org/history-harga/{year}/{month_str}/{day}/"

    if debug:
        print(f"[DEBUG] Mengakses URL: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Melempar error jika status != 200
    except requests.RequestException as e:
        if debug:
            print(f"[DEBUG] Gagal mengakses {url} karena: {e}")
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Mencari tabel di halaman
    tables = soup.find_all('table')
    if len(tables) < 2:
        if debug:
            print("[DEBUG] Tidak menemukan minimal 2 <table> pada halaman.")
        return None, None

    # Biasanya table ke-2 (index 1) menampilkan harga emas
    second_table = tables[1]
    rows = second_table.find_all('tr')

    # Indeks baris/kolom (tergantung struktur HTML situs)
    usd_index = (2, 1)  # (row, col)
    idr_index = (3, 3)  # (row, col)

    try:
        usd_value = rows[usd_index[0]].find_all('td')[usd_index[1]].get_text(strip=True)
        idr_value = rows[idr_index[0]].find_all('td')[idr_index[1]].get_text(strip=True)

        if debug:
            print(f"[DEBUG] Berhasil scrape: USD={usd_value}, IDR={idr_value}")

        return usd_value, idr_value
    except (IndexError, AttributeError):
        if debug:
            print("[DEBUG] Struktur tabel tidak sesuai (IndexError/AttributeError).")
        return None, None


def scrape_year(year: int, overall_start: date, overall_end: date, debug: bool = False):
    """
    Scrape data untuk satu tahun (year), dibatasi oleh overall_start dan overall_end.
    Return: list of dict, yang tiap dict berisi {Date, USD, IDR}.

    Param:
      - debug: jika True, akan menampilkan info URL, hasil scraping per hari, dsb.
    """
    # Buat start date & end date untuk tahun "year"
    year_start = date(year, 1, 1)
    year_end   = date(year, 12, 31)

    # Pastikan tidak keluar dari overall range
    actual_start = max(year_start, overall_start)
    actual_end   = min(year_end, overall_end)

    # Jika intervalnya tidak valid, return list kosong
    if actual_start > overall_end or actual_end < overall_start:
        return []

    print(f"\n[INFO] Mulai scraping tahun {year}, range: {actual_start} s/d {actual_end}")

    data_year = []
    current_date = actual_start

    # Hitung total hari untuk tahun ini (digunakan untuk tqdm)
    total_days = (actual_end - actual_start).days + 1

    # Gunakan tqdm untuk menampilkan progress bar harian
    with tqdm(total=total_days, desc=f"Scraping Progress (Year {year})") as pbar:
        while current_date <= actual_end:
            y  = current_date.year
            m  = MONTH_NAMES[current_date.month]
            d  = current_date.day

            usd, idr = scrape_gold_price(y, m, d, debug=debug)

            data_year.append({
                "Date": current_date.isoformat(),  # YYYY-MM-DD
                "USD":  usd,
                "IDR":  idr
            })

            # (Opsional) jeda agar tidak terlalu cepat
            # time.sleep(1)

            current_date += timedelta(days=1)
            pbar.update(1)  # Update progress bar

    print(f"[INFO] Selesai scraping tahun {year}. Total data: {len(data_year)}")

    return data_year


def main():
    # Batas total scraping
    overall_start = date(2013, 9, 1)   # Mulai: 1 September 2013
    overall_end   = date(2025, 1, 18)  # Hingga: 18 Januari 2025

    all_data = []

    # Loop tahun 2013 s/d 2025
    for year in range(2013, 2026):
        # Di sini kita set debug=False agar tidak terlalu banyak print
        # Jika butuh info lebih detail, ubah jadi debug=True.
        data_for_year = scrape_year(year, overall_start, overall_end, debug=False)

        all_data.extend(data_for_year)  # Gabungkan ke list utama

        # Pesan bahwa tahun ini sudah selesai
        print(f"-> [INFO] Tahun {year} selesai. Data ditambahkan: {len(data_for_year)}")

    # Setelah seluruh tahun digabung, simpan ke CSV
    csv_filename = "harga_emas_2013_2025_volatility.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Date", "USD", "IDR"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\n[INFO] Berhasil menyimpan total {len(all_data)} baris data ke '{csv_filename}'.")


if __name__ == "__main__":
    main()
