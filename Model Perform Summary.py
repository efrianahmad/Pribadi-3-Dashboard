# ============================================================
# POWER BI PYTHON VISUAL 3 — MODEL SUMMARY & FEATURE IMPORTANCE
# Kolom yang di-drag ke Fields:
#   - Tanggal_Murni, penjualan
# Size: ~960x540
# ============================================================
import pandas as pd, numpy as np, warnings
import matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    (3,  1, 19, 'ramadhan'),   # 1–15 Maret (masih Ramadhan)

    # Menjelang + Hari Raya Idul Fitri: 16–25 Maret
    # (16–18 Mar cuti menjelang, 19 Mar Nyepi, 20–21 Mar Lebaran,
    #  22–25 Mar cuti bersama estimasi)
    (3, 19, 22, 'libur'),

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
    'peak':       '📚 Peak Study Tour',
    'normal':     '📅 Normal',
    'uas':        '📝 UAS / Penilaian Akhir',
    'libur':      '🏖️ Libur Panjang',
    'agustusan':  '🇮🇩 17 Agustus',
    'uts':        '📝 UTS / TKA / Penilaian',
    'ramadhan':   '🕌 Ramadhan / Tutup',
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
ct = [c for c in df.columns if 'tanggal' in c or 'date' in c][0]
cp = [c for c in df.columns if 'penjualan' in c or 'omzet' in c or 'sales' in c][0]
df['tgl'] = pd.to_datetime(df[ct])
df['val'] = pd.to_numeric(df[cp], errors='coerce')
df = df.dropna(subset=['tgl','val'])
df['b']   = df['tgl'].dt.month
df['h']   = df['tgl'].dt.day
df['hw']  = df['tgl'].dt.dayofweek
df['isw'] = (df['hw']>=5).astype(int)
df['ms']  = df.apply(lambda r: get_musim(r['b'],r['h']), axis=1)
df['msc'] = df['ms'].map(MUSIM_SCORE)
df['ip']  = (df['ms']=='peak').astype(int)
df['iu']  = (df['ms']=='uts').astype(int)
df['ia']  = (df['ms']=='uas').astype(int)
df['il']  = (df['ms']=='libur').astype(int)
df['ir']  = (df['ms']=='ramadhan').astype(int)
df['ig']  = (df['ms']=='agustusan').astype(int)
df['ir']  = (df['ms']=='ramadhan').astype(int)
df['ig']  = (df['ms']=='agustusan').astype(int)

df['wk'] = df['tgl'].dt.to_period('W').apply(lambda r: r.start_time)
W = df.groupby('wk').agg(
    total=('val','sum'), b=('b','first'), h=('h','first'),
    msc=('msc','mean'), ip=('ip','mean'), iu=('iu','mean'),
    ia=('ia','mean'), il=('il','mean'),
    ir=('ir','mean'), ig=('ig','mean'), ms=('ms','first'),
).reset_index().sort_values('wk').reset_index(drop=True)
W['i']  = range(len(W))
W['sw'] = np.sin(2*np.pi*W['i']/4)
W['cw'] = np.cos(2*np.pi*W['i']/4)
W['sm'] = np.sin(2*np.pi*W['b']/12)
W['cm'] = np.cos(2*np.pi*W['b']/12)
W['l1'] = W['total'].shift(1)
W['l2'] = W['total'].shift(2)
W['l3'] = W['total'].shift(1).rolling(3).mean()
W = W.dropna().reset_index(drop=True)

FT    = ['i','sw','cw','sm','cm','msc','ip','iu','ia','il','ir','ig','l1','l2','l3']
FLBL  = {
    'i':'Minggu ke-','sw':'Sin(Minggu)','cw':'Cos(Minggu)',
    'sm':'Sin(Bulan)','cm':'Cos(Bulan)',
    'msc':'Skor Musim','ip':'Peak Study Tour',
    'iu':'Musim UTS','ia':'Musim UAS','il':'Libur Panjang',
    'ir':'Ramadhan/Tutup','ig':'17 Agustus',
    'l1':'Omzet Minggu Lalu','l2':'Omzet 2 Minggu Lalu',
    'l3':'Rolling Avg 3 Minggu',
}
FCOLOR = {
    'msc':'#107C10','ip':'#107C10','iu':'#D83B01',
    'ia':'#FF8C00','il':'#5C2D91',
    'ir':'#004E8C','ig':'#E3008C',
    'l1':'#118DFF','l2':'#118DFF','l3':'#118DFF',
}

X = W[FT]; y = W['total']
n_te = min(5, len(W)//4)
Xtr, Xte = X.iloc[:-n_te], X.iloc[-n_te:]
ytr, yte = y.iloc[:-n_te], y.iloc[-n_te:]
sc = StandardScaler()
Xtr_sc = sc.fit_transform(Xtr)
Xte_sc = sc.transform(Xte)

gb  = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
rf  = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42)
rid = Ridge(alpha=1.0)
gb.fit(Xtr, ytr);   pred_gb  = np.clip(gb.predict(Xte), 0, None)
rf.fit(Xtr, ytr);   pred_rf  = np.clip(rf.predict(Xte), 0, None)
rid.fit(Xtr_sc, ytr); pred_rid = np.clip(rid.predict(Xte_sc), 0, None)

def metrics(yt, yp):
    return {'r2':r2_score(yt,yp),
            'mae':mean_absolute_error(yt,yp),
            'mape':mean_absolute_percentage_error(yt,yp)*100}

M = {
    'Gradient Boosting': metrics(yte, pred_gb),
    'Random Forest':     metrics(yte, pred_rf),
    'Ridge Regression':  metrics(yte, pred_rid),
}
best = max(M, key=lambda k: M[k]['r2'])

# Feature importance (GB)
fi = pd.Series(gb.feature_importances_, index=FT).sort_values(ascending=True)

# ── PLOT ────────────────────────────────────────────────────
BG = '#FFFFFF'; FONT = '#252423'; GRAY = '#A0A0A0'; GRID_C = '#E5E5E5'

fig = plt.figure(figsize=(19.2, 9.5))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                        left=0.05, right=0.98, top=0.88, bottom=0.08)

# ── 1. Feature Importance ───────────────────────────────────
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_facecolor(BG)
colors_fi = [FCOLOR.get(f, GRAY) for f in fi.index]
bars = ax1.barh([FLBL[f] for f in fi.index], fi.values,
                color=colors_fi, edgecolor=BG, height=0.65)
ax1.set_xlabel('Importance Score', fontsize=9, color=FONT)
ax1.set_title('Feature Importance\n(Gradient Boosting)',
              fontsize=11, fontweight='bold', color=FONT, pad=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_color(GRID_C)
ax1.tick_params(colors=FONT, labelsize=8.5, length=0)
ax1.grid(axis='x', color=GRID_C, lw=0.8)
ax1.set_axisbelow(True)
for bar, val in zip(bars, fi.values):
    ax1.text(val + 0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8, color=FONT)

# Legend fi
leg_items = [
    mpatches.Patch(color='#107C10', label='Musim Sekolah (baru)'),
    mpatches.Patch(color='#118DFF', label='Lag / Trend'),
    mpatches.Patch(color=GRAY,      label='Fitur Waktu'),
]
ax1.legend(handles=leg_items, fontsize=8, frameon=False,
           loc='lower right')

# ── 2. Actual vs Predicted (Best model) ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(BG)
best_pred = {'Gradient Boosting':pred_gb,
             'Random Forest':pred_rf,
             'Ridge Regression':pred_rid}[best]
x_pos = range(len(yte))
ax2.plot(x_pos, yte.values/1e6, color='#118DFF', lw=2.2,
         marker='o', ms=6, label='Aktual', zorder=3)
ax2.plot(x_pos, best_pred/1e6, color='#107C10', lw=2.2,
         marker='s', ms=6, ls='--', label=f'Prediksi ({best})', zorder=3)
ax2.fill_between(x_pos, yte.values/1e6, best_pred/1e6,
                 alpha=0.1, color='#107C10')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'Mg {i+1}' for i in x_pos], fontsize=8.5)
ax2.set_ylabel('Omzet Mingguan (Juta Rp)', fontsize=9)
ax2.set_title(f'Aktual vs Prediksi — {n_te} Minggu Terakhir',
              fontsize=11, fontweight='bold', color=FONT)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(GRID_C)
ax2.spines['bottom'].set_color(GRID_C)
ax2.tick_params(colors=FONT, labelsize=8.5)
ax2.grid(axis='y', color=GRID_C, lw=0.8)
ax2.set_axisbelow(True)
ax2.legend(fontsize=8.5, frameon=False)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'Rp{x:.0f}M'))

# ── 3. R² Comparison ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(BG)
mn = list(M.keys())
r2v= [M[m]['r2'] for m in mn]
col= ['#107C10' if m==best else '#118DFF' for m in mn]
bars3 = ax3.bar(['GB','RF','Ridge'], r2v, color=col,
                edgecolor=BG, width=0.5)
ax3.axhline(0, color=FONT, lw=1)
ax3.set_title('R² Score per Model\n(lebih tinggi = lebih baik)',
              fontsize=11, fontweight='bold', color=FONT)
ax3.set_ylabel('R² Score', fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(GRID_C)
ax3.spines['bottom'].set_color(GRID_C)
ax3.tick_params(colors=FONT, labelsize=9)
ax3.grid(axis='y', color=GRID_C, lw=0.8)
ax3.set_axisbelow(True)
for bar, val in zip(bars3, r2v):
    ax3.text(bar.get_x()+bar.get_width()/2,
             max(val,0)+0.01, f'{val:.3f}',
             ha='center', fontsize=10, fontweight='bold', color=FONT)

# ── 4. MAPE Comparison ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG)
mapev = [M[m]['mape'] for m in mn]
col4  = ['#107C10' if m==best else '#118DFF' for m in mn]
bars4 = ax4.bar(['GB','RF','Ridge'], mapev, color=col4,
                edgecolor=BG, width=0.5)
ax4.set_title('MAPE per Model\n(lebih rendah = lebih baik)',
              fontsize=11, fontweight='bold', color=FONT)
ax4.set_ylabel('MAPE (%)', fontsize=9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color(GRID_C)
ax4.spines['bottom'].set_color(GRID_C)
ax4.tick_params(colors=FONT, labelsize=9)
ax4.grid(axis='y', color=GRID_C, lw=0.8)
ax4.set_axisbelow(True)
for bar, val in zip(bars4, mapev):
    ax4.text(bar.get_x()+bar.get_width()/2,
             val+0.3, f'{val:.1f}%',
             ha='center', fontsize=10, fontweight='bold', color=FONT)

# ── 5. MAE Comparison ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(BG)
maev = [M[m]['mae']/1e6 for m in mn]
col5 = ['#107C10' if m==best else '#118DFF' for m in mn]
bars5 = ax5.bar(['GB','RF','Ridge'], maev, color=col5,
                edgecolor=BG, width=0.5)
ax5.set_title('MAE per Model\n(lebih rendah = lebih baik)',
              fontsize=11, fontweight='bold', color=FONT)
ax5.set_ylabel('MAE (Juta Rp)', fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_color(GRID_C)
ax5.spines['bottom'].set_color(GRID_C)
ax5.tick_params(colors=FONT, labelsize=9)
ax5.grid(axis='y', color=GRID_C, lw=0.8)
ax5.set_axisbelow(True)
for bar, val in zip(bars5, maev):
    ax5.text(bar.get_x()+bar.get_width()/2,
             val+0.01, f'Rp{val:.2f}M',
             ha='center', fontsize=10, fontweight='bold', color=FONT)

# ── Header ───────────────────────────────────────────────────
'''fig.text(0.01, 0.97,
         'Model Performance Summary — QRIS Predictive Analytics',
         fontsize=13, fontweight='bold', color=FONT, va='top')
fig.text(0.01, 0.92,
         f'Target: Omzet Mingguan  |  Best Model: {best}  |  '
         f"R²={M[best]['r2']:.3f}  |  MAPE={M[best]['mape']:.1f}%  |  "
         f"MAE=Rp{M[best]['mae']/1e6:.2f}M",
         fontsize=9, color=GRAY, va='top')'''

plt.savefig('D:\\Belajar-Belajar (non-Kuliah)\\Predictive Pribadi 3\\pbi_v3_model.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
