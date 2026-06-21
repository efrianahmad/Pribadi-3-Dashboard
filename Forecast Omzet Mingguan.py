# ============================================================
# POWER BI PYTHON VISUAL 1 — FORECAST OMZET MINGGUAN
# Kolom yang di-drag ke Fields:
#   - Tanggal_Murni   (Date)
#   - penjualan       (Numeric)
# Canvas: 1920x1080, taruh visual ini ~960x540 (setengah halaman atas)
# ============================================================
import pandas as pd, numpy as np, warnings
import matplotlib, matplotlib.pyplot as plt, matplotlib.patches as mpatches
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

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


# ── Baca dataset dari Power BI ──────────────────────────────
df = dataset.copy()
df.columns = [c.lower().strip() for c in df.columns]
ct = [c for c in df.columns if 'tanggal' in c or 'date' in c][0]
cp = [c for c in df.columns if 'penjualan' in c or 'omzet' in c or 'sales' in c][0]
df['tgl'] = pd.to_datetime(df[ct])
df['val'] = pd.to_numeric(df[cp], errors='coerce')
df = df.dropna(subset=['tgl','val'])
df['b']  = df['tgl'].dt.month
df['h']  = df['tgl'].dt.day
df['hw'] = df['tgl'].dt.dayofweek
df['ms'] = df.apply(lambda r: get_musim(r['b'],r['h']), axis=1)
df['msc']= df['ms'].map(MUSIM_SCORE)

# ── Weekly aggregation ──────────────────────────────────────
df['wk'] = df['tgl'].dt.to_period('W').apply(lambda r: r.start_time)
W = df.groupby('wk').agg(
    total=('val','sum'), cnt=('val','count'),
    b=('b','first'), h=('h','first'),
    msc=('msc','mean'),
    is_peak=('ms', lambda x: (x=='peak').mean()),
    is_uts =('ms', lambda x: (x=='uts').mean()),
    is_uas =('ms', lambda x: (x=='uas').mean()),
    is_lib =('ms', lambda x: (x=='libur').mean()),
    is_ram =('ms', lambda x: (x=='ramadhan').mean()),
    is_agu =('ms', lambda x: (x=='agustusan').mean()),
    ms=('ms','first'),
).reset_index().sort_values('wk').reset_index(drop=True)
W['i']   = range(len(W))
W['sw']  = np.sin(2*np.pi*W['i']/4)
W['cw']  = np.cos(2*np.pi*W['i']/4)
W['sm']  = np.sin(2*np.pi*W['b']/12)
W['cm']  = np.cos(2*np.pi*W['b']/12)
W['l1']  = W['total'].shift(1)
W['l2']  = W['total'].shift(2)
W['l3']  = W['total'].shift(1).rolling(3).mean()
W = W.dropna().reset_index(drop=True)

FT = ['i','sw','cw','sm','cm','msc','is_peak','is_uts','is_uas','is_lib','is_ram','is_agu','l1','l2','l3']
X  = W[FT]; y = W['total']
n_te = min(5, len(W)//4)
Xtr, Xte = X.iloc[:-n_te], X.iloc[-n_te:]
ytr, yte = y.iloc[:-n_te], y.iloc[-n_te:]

sc  = StandardScaler()
gb  = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
rid = Ridge(alpha=1.0)
gb.fit(Xtr, ytr)
rid.fit(sc.fit_transform(Xtr), ytr)

pred_te_gb  = np.clip(gb.predict(Xte), 0, None)
pred_te_rid = np.clip(rid.predict(sc.transform(Xte)), 0, None)
r2_gb   = r2_score(yte, pred_te_gb)
mape_gb = mean_absolute_percentage_error(yte, pred_te_gb)*100
mae_gb  = mean_absolute_error(yte, pred_te_gb)

# ── Forecast 8 minggu ke depan ──────────────────────────────
last_i  = W['i'].max()
last_wk = W['wk'].max()
last_l1 = W['total'].iloc[-1]
last_l2 = W['total'].iloc[-2]
last_l3 = W['total'].iloc[-3:].mean()

fut = []
l1, l2, l3_vals = last_l1, last_l2, list(W['total'].iloc[-2:].values)
for k in range(8):
    i   = last_i + k + 1
    dt  = last_wk + pd.Timedelta(weeks=k+1)
    b, h = dt.month, dt.day
    ms  = get_musim(b, h)
    msc = MUSIM_SCORE[ms]
    l3  = np.mean(l3_vals[-3:]) if len(l3_vals)>=3 else np.mean(l3_vals)
    row = {'i':i,'sw':np.sin(2*np.pi*i/4),'cw':np.cos(2*np.pi*i/4),
           'sm':np.sin(2*np.pi*b/12),'cm':np.cos(2*np.pi*b/12),
           'msc':msc,'is_peak':1 if ms=='peak' else 0,
           'is_uts':1 if ms=='uts' else 0,'is_uas':1 if ms=='uas' else 0,
           'is_lib':1 if ms=='libur' else 0,
           'is_ram':1 if ms=='ramadhan' else 0,
           'is_agu':1 if ms=='agustusan' else 0,
           'l1':l1,'l2':l2,'l3':l3,
           'dt':dt,'ms':ms}
    fut.append(row)
    # update lags
    p = float(np.clip(gb.predict(pd.DataFrame([row])[FT]), 0, None)[0])
    l2 = l1; l1 = p
    l3_vals.append(p)

FD  = pd.DataFrame(fut)
FT2 = FD[FT]
pf_gb  = np.clip(gb.predict(FT2), 0, None)
pf_rid = np.clip(rid.predict(sc.transform(FT2)), 0, None)

# Bootstrap CI
boots = []
for _ in range(150):
    idx = np.random.choice(len(Xtr), len(Xtr), replace=True)
    m = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                   learning_rate=0.1, random_state=None)
    m.fit(Xtr.iloc[idx], ytr.iloc[idx])
    boots.append(np.clip(m.predict(FT2), 0, None))
boots  = np.array(boots)
ci_lo  = np.clip(np.percentile(boots, 15, axis=0), 0, None)
ci_hi  = np.percentile(boots, 85, axis=0)

# ── PLOT ────────────────────────────────────────────────────
# Styling mirip Power BI
BG   = '#FFFFFF'; GRID = '#E5E5E5'; FONT = '#252423'
BLUE = '#118DFF'; GRAY = '#A0A0A0'

fig, ax = plt.subplots(figsize=(19.2, 5.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Aktual
ax.fill_between(W['wk'], W['total']/1e6, alpha=0.12, color=BLUE)
ax.plot(W['wk'], W['total']/1e6, color=BLUE, lw=2.2,
        marker='o', ms=5, label='Aktual', zorder=3)

# Forecast dengan warna musim
for k in range(len(FD)):
    col = MUSIM_COLOR[FD['ms'].iloc[k]]
    if k < len(FD)-1:
        ax.plot([FD['dt'].iloc[k], FD['dt'].iloc[k+1]],
                [pf_gb[k]/1e6, pf_gb[k+1]/1e6],
                color=col, lw=2.2, alpha=0.9)
ax.scatter(FD['dt'], pf_gb/1e6,
           c=[MUSIM_COLOR[m] for m in FD['ms']], s=80,
           zorder=5, edgecolors=BG, lw=1.5)

# CI band
ax.fill_between(FD['dt'], ci_lo/1e6, ci_hi/1e6,
                alpha=0.12, color=GRAY)

# Garis batas
ax.axvline(last_wk, color='#D83B01', lw=1.5, ls='--', alpha=0.7)

# Label nilai forecast
for k in range(len(FD)):
    ax.annotate(f"Rp{pf_gb[k]/1e6:.1f}M",
                xy=(FD['dt'].iloc[k], pf_gb[k]/1e6),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=7.5, color=MUSIM_COLOR[FD['ms'].iloc[k]],
                fontweight='bold')

# Styling
ax.set_facecolor(BG)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=FONT, labelsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'Rp{x:.0f}M'))
ax.grid(axis='y', color=GRID, linewidth=0.8, linestyle='-')
ax.set_axisbelow(True)
ax.set_xlabel('')

# Title & subtitle
fig.text(0.01, 0.97, 'Forecast Omzet Mingguan — 8 Minggu ke Depan',
         fontsize=13, fontweight='bold', color=FONT, va='top')
fig.text(0.01, 0.88,
         f'Gradient Boosting  |  R² = {r2_gb:.3f}  |  MAE = Rp{mae_gb/1e6:.2f}M  |  MAPE = {mape_gb:.1f}%',
         fontsize=9, color=GRAY, va='top')

# Legend musim
patches = [mpatches.Patch(color=MUSIM_COLOR[s], label=MUSIM_LABEL[s]) for s in MUSIM_LABEL]
ax.legend(handles=patches, fontsize=8.5, frameon=False,
          loc='upper left', ncol=5,
          bbox_to_anchor=(0, 0.82), bbox_transform=fig.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.80])
plt.savefig('D:\\Belajar-Belajar (non-Kuliah)\\Predictive Pribadi 3\\pbi_v1_forecast.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
