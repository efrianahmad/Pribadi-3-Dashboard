# ============================================================
# POWER BI PYTHON VISUAL 2 — HEATMAP PREDIKSI JAM x HARI
# Kolom yang di-drag ke Fields:
#   - Tanggal_Murni, penjualan, Jam_Angka, Nama_Hari
# Size: ~960x540
# ============================================================
import pandas as pd, numpy as np, warnings
import matplotlib, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# KONFIGURASI MUSIM SEKOLAH — UPDATE PER TAHUN AJARAN
# Sumber: Kalender Akademik & Hari Libur Nasional Indonesia
# Terakhir diupdate: Tahun Ajaran 2025/2026
#
# CARA EDIT:
#   Hanya ubah bagian JADWAL_MUSIM di bawah ini.
#   Format: (bulan, hari_mulai, hari_selesai, 'kode_musim')
#   Urutan penting! Kondisi pertama yang cocok dipakai.
# ============================================================

JADWAL_MUSIM = [

    # ════════════════════════════════════════════════════════
    # RAMADHAN & LIBUR LEBARAN
    # Update setiap tahun (bergeser ~11 hari/tahun)
    # ════════════════════════════════════════════════════════
    # Ramadhan 2026: 16 Feb – 19 Mar
    # (Libur awal puasa 16–20 Feb, lanjut Ramadhan sampai 19 Mar)
    (2, 16, 28, 'ramadhan'),   # 16–28 Februari
    (3,  1, 19, 'ramadhan'),   # 1–19 Maret (masih Ramadhan)

    # Menjelang + Hari Raya Idul Fitri: 16–25 Maret
    # (16–18 Mar cuti menjelang, 19 Mar Nyepi, 20–21 Mar Lebaran,
    #  22–25 Mar cuti bersama estimasi)
    (3, 16, 22, 'libur'),

    # Sisa Maret setelah Lebaran: 26–31 Mar
    # Masih sepi, baru mulai recovery — anggap normal dulu
    (3, 26, 31, 'normal'),

    # Ramadhan 2027: ~6 Feb – 7 Mar (perkiraan, uncomment nanti)
    # (2,  6, 28, 'ramadhan'),
    # (3,  1,  7, 'ramadhan'),
    # (3,  8, 18, 'libur'),    # Lebaran 2027

    # ════════════════════════════════════════════════════════
    # JANUARI — LIBUR SEMESTER GANJIL
    # ════════════════════════════════════════════════════════
    (1,  1,  7, 'libur'),     # Libur semester + Tahun Baru

    # ════════════════════════════════════════════════════════
    # APRIL — PENILAIAN & UJIAN (TKA)
    # Penilaian Semester Kelas Akhir: 13–17 Apr
    # TKA SMP: 6–16 Apr → study tour sepi selama ujian
    # TKA SD: 20–30 Apr → masih sepi
    # ════════════════════════════════════════════════════════
    (4,  6, 30, 'uts'),       # Seluruh April periode ujian

    # ════════════════════════════════════════════════════════
    # MEI — PENILAIAN SEMESTER AKHIR
    # Penilaian 4–8 Mei dan 18–22 Mei
    # Di luar tanggal itu masih bisa ada study tour tapi berkurang
    # ════════════════════════════════════════════════════════
    (5,  1, 10, 'uts'),       # Awal Mei: penilaian + Hari Buruh
    (5, 11, 17, 'peak'),      # 11–17 Mei: jeda penilaian, masih bisa tour
    (5, 18, 31, 'uts'),       # 18–31 Mei: penilaian akhir + Idul Adha

    # ════════════════════════════════════════════════════════
    # JUNI — UAS + LIBUR KENAIKAN KELAS
    # ════════════════════════════════════════════════════════
    (6,  1, 19, 'uas'),       # 1–19 Juni: UAS + pembagian rapor
    (6, 20, 30, 'libur'),     # 20–30 Juni: libur kenaikan kelas

    # ════════════════════════════════════════════════════════
    # JULI — LIBUR + AWAL TAHUN AJARAN BARU
    # Tahun ajaran baru mulai 13 Juli
    # ════════════════════════════════════════════════════════
    (7,  1, 12, 'libur'),     # 1–12 Juli: masih libur
    (7, 13, 31, 'normal'),    # 13–31 Juli: masuk sekolah, belum ada tour

    # ════════════════════════════════════════════════════════
    # AGUSTUS — PERIODE 17 AGUSTUS
    # Sepi karena lomba, pawai, upacara kemerdekaan
    # Agustus awal (1–9) masih bisa ada study tour
    # ════════════════════════════════════════════════════════
    (8,  1,  9, 'peak'),      # 1–9 Agustus: masih bisa study tour
    (8, 10, 31, 'agustusan'), # 10–31 Agustus: persiapan + pasca 17an

    # ════════════════════════════════════════════════════════
    # OKTOBER — UTS SEMESTER GANJIL
    # ════════════════════════════════════════════════════════
    (10, 1, 31, 'uts'),       # Seluruh Oktober: UTS semester ganjil

    # ════════════════════════════════════════════════════════
    # DESEMBER — UAS + LIBUR SEMESTER GANJIL
    # ════════════════════════════════════════════════════════
    (12,  1, 15, 'uas'),      # 1–15 Desember: UAS semester ganjil
    (12, 16, 31, 'libur'),    # 16–31 Desember: libur semester
]

# ── Skor pengaruh ke model ───────────────────────────────────
# Makin tinggi = makin ramai / bagus untuk bisnis
MUSIM_SCORE = {
    'peak':       2,      # Study tour ramai
    'normal':     1,      # Hari biasa
    'uas':       -1,      # Ujian akhir semester
    'libur':     -1.5,    # Libur panjang (ada tapi sepi)
    'agustusan': -1.5,    # Sekitar 17 Agustus
    'uts':       -2,      # Ujian tengah semester / TKA
    'ramadhan':  -3,      # Toko tutup total
}

# ── Warna untuk chart ────────────────────────────────────────
MUSIM_COLOR = {
    'peak':       '#107C10',   # Hijau
    'normal':     '#0078D4',   # Biru
    'uas':        '#FF8C00',   # Oranye
    'libur':      '#5C2D91',   # Ungu
    'agustusan':  '#E3008C',   # Pink
    'uts':        '#D83B01',   # Merah
    'ramadhan':   '#004E8C',   # Biru tua
}

# ── Label untuk legend ───────────────────────────────────────
MUSIM_LABEL = {
    'peak':       'Peak Study Tour',
    'normal':     'Normal',
    'uas':        'UAS / Penilaian Akhir',
    'libur':      'Libur Panjang',
    'agustusan':  '17 Agustus',
    'uts':        'UTS / TKA / Penilaian',
    'ramadhan':   'Ramadhan / Tutup',
}

def get_musim(b, h):
    """
    Cek musim berdasarkan bulan (b) dan hari (h).
    Membaca JADWAL_MUSIM dari atas ke bawah —
    kondisi pertama yang cocok langsung dipakai.
    Kalau tidak ada yang cocok, cek apakah Peak Tour atau Normal.
    """
    for (bln, h_mulai, h_selesai, musim) in JADWAL_MUSIM:
        if b == bln and h_mulai <= h <= h_selesai:
            return musim

    # Peak Study Tour — bulan yang tidak masuk jadwal di atas
    # Feb awal (sebelum Ramadhan), Sep, Nov = peak tour
    if b == 2 and h <= 15:   return 'peak'   # Feb sebelum Ramadhan
    if b == 9:                return 'peak'   # September
    if b == 11:               return 'peak'   # November

    return 'normal'


df = dataset.copy()
df.columns = [c.lower().strip() for c in df.columns]
ct  = [c for c in df.columns if 'tanggal' in c or 'date' in c][0]
cp  = [c for c in df.columns if 'penjualan' in c or 'omzet' in c or 'sales' in c][0]
cj_candidates = [c for c in df.columns if 'jam' in c or 'hour' in c]
# pilih kolom jam yang berisi angka (bukan waktu string)
cj = None
for cc in cj_candidates:
    try:
        test = pd.to_numeric(df[cc], errors='coerce').dropna()
        if len(test) > 0 and test.max() <= 23 and test.min() >= 0:
            cj = cc; break
    except: pass
if cj is None: cj = cj_candidates[0]
df['tgl'] = pd.to_datetime(df[ct])
df['val'] = pd.to_numeric(df[cp], errors='coerce')
df['jam'] = pd.to_numeric(df[cj], errors='coerce').astype('Int64')
df['b']   = df['tgl'].dt.month
df['h']   = df['tgl'].dt.day
df['hw']  = df['tgl'].dt.dayofweek
df['isw'] = (df['hw'] >= 5).astype(int)
df['ms']  = df.apply(lambda r: get_musim(r['b'],r['h']), axis=1)
df['msc'] = df['ms'].map(MUSIM_SCORE)
df = df.dropna(subset=['val','jam'])
df['jam'] = df['jam'].astype(int)

def sesi(j):
    if j<6: return 0
    elif j<12: return 1
    elif j<17: return 2
    elif j<22: return 3
    return 0

df['sesi']   = df['jam'].apply(sesi)
df['ispeak'] = ((df['sesi']==3)).astype(int)

FEAT = ['jam','hw','isw','sesi','ispeak','b','h','msc']
rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                            min_samples_leaf=15, random_state=42, n_jobs=-1)
rf.fit(df[FEAT], df['val'])

# Grid prediksi untuk 3 skenario musim
grids = {}
for musim_nm, bln, hari_ref, msc_val in [
    ('Peak Study Tour\n(Mei)', 5, 15, 2),
    ('Normal (November)', 11, 15, 1),
    ('Musim UTS\n(Maret)', 3, 10, -2),
]:
    g = np.zeros((24, 7))
    for hw in range(7):
        for j in range(24):
            row = pd.DataFrame([{
                'jam':j,'hw':hw,'isw':1 if hw>=5 else 0,
                'sesi':sesi(j),'ispeak':1 if 17<=j<=21 else 0,
                'b':bln,'h':hari_ref,'msc':msc_val
            }])
            g[j, hw] = rf.predict(row)[0]
    grids[musim_nm] = g

# ── PLOT ────────────────────────────────────────────────────
BG = '#FFFFFF'; FONT = '#252423'; GRID_C = '#E5E5E5'
HCOLS = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
JAM   = [f'{j:02d}:00' for j in range(24)]

fig, axes = plt.subplots(1, 3, figsize=(19.2, 9.5))
fig.patch.set_facecolor(BG)

cmaps   = ['YlGn', 'Blues', 'OrRd']
titles  = list(grids.keys())
clabels = ['Peak Study Tour', 'Normal', 'Musim UTS']

for ax, (title, grid), cmap, clabel in zip(
        axes, grids.items(), cmaps, clabels):
    ax.set_facecolor(BG)
    data = grid / 1000  # ke ribu Rp

    # Custom colormap
    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   interpolation='nearest')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax.grid(which='minor', color=BG, linewidth=1.5)

    # Annotations
    vmin, vmax = data.min(), data.max()
    for j in range(24):
        for h in range(7):
            val = data[j, h]
            # warna teks: putih kalau background gelap
            norm_val = (val - vmin) / (vmax - vmin + 1e-9)
            txt_color = 'white' if norm_val > 0.55 else '#252423'
            txt = f'{val:.0f}K' if val >= 1 else f'{val*1000:.0f}'
            ax.text(h, j, txt, ha='center', va='center',
                    fontsize=6.5, color=txt_color, fontweight='bold')

    ax.set_xticks(range(7))
    ax.set_xticklabels(HCOLS, fontsize=9, color=FONT)
    ax.set_yticks(range(24))
    ax.set_yticklabels(JAM, fontsize=7.5, color=FONT)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    # Title
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=FONT, pad=12)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Prediksi Nilai Transaksi (Ribu Rp)',
                 fontsize=8, color=FONT)
    cb.ax.tick_params(labelsize=8, colors=FONT)
    cb.outline.set_visible(False)

# Highlight peak hours (jam 17-21)
for ax in axes:
    ax.axhline(16.5, color='#D83B01', lw=1.5, ls='--', alpha=0.6)
    ax.axhline(21.5, color='#D83B01', lw=1.5, ls='--', alpha=0.6)
    ax.text(6.7, 19, '← Peak\n   17-21', fontsize=7,
            color='#D83B01', va='center', fontweight='bold')

fig.text(0.01, 0.98,
         'Prediksi Nilai Transaksi per Jam & Hari (Ribu Rp)',
         fontsize=13, fontweight='bold', color=FONT, va='top')
fig.text(0.01, 0.93,
         'Random Forest  |  Tiga skenario musim sekolah  |  '
         'Nilai = rata-rata prediksi per transaksi',
         fontsize=9, color='#A0A0A0', va='top')

plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig('D:\\Belajar-Belajar (non-Kuliah)\\Predictive Pribadi 3\\pbi_v2_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
