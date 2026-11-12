import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Dashboard Perencanaan Produksi dan Biaya UMKM", layout="wide")

# ---------- Data contoh (SIMULASI, ganti dengan data usaha sebenarnya) ----------
SALES0 = pd.DataFrame({
    "bulan": [f"M{i}" for i in range(1, 13)],
    "permintaan": [120, 132, 128, 140, 151, 149, 160, 172, 168, 181, 190, 185],
})
BOM0 = pd.DataFrame({
    "bahan": ["Bahan baku utama", "Bahan penolong", "Kemasan"],
    "satuan": ["kg", "kg", "pcs"],
    "per_unit": [0.30, 0.10, 1.0],
    "harga": [15000, 25000, 800],
    "stok_awal": [50.0, 20.0, 100.0],
})
PAR0 = dict(tarif=25000, waktu=42.0, oh_tetap=1500000, oh_var=500, yield_=95,
            markup=40, jam=160, stok_produk=30, stok_aman=20, horizon=3,
            holdout=3, interp=False, aman_mode="Manual", service=95,
            harga_mode="Markup dari HPP", harga_manual=40000)

st.session_state.setdefault("sales", SALES0.copy())
st.session_state.setdefault("bom", BOM0.copy())
st.session_state.setdefault("par", dict(PAR0))
st.session_state.setdefault("cfg", (3, 0.3, 0.5, 0.2))  # n, alpha, alpha Holt, beta Holt
st.session_state.setdefault("ver", 0)  # versi editor, dinaikkan saat CSV baru dimuat
for k, v in PAR0.items():  # lengkapi parameter jika session lama belum punya kunci baru
    st.session_state.par.setdefault(k, v)


def rp(x):
    return "Rp{:,.0f}".format(x).replace(",", ".")


# =====================================================================
# Bagian hitung (tidak memakai Streamlit, supaya mudah diuji)
# =====================================================================
def f_ma(y, n):
    f = [np.nan] * n + [y[i - n:i].mean() for i in range(n, len(y))]
    return np.array(f), (lambda h: np.full(h, y[-n:].mean()))


def f_ses(y, a):
    lvl, f = y[0], [np.nan]
    for t in range(1, len(y)):
        f.append(lvl)
        lvl = a * y[t] + (1 - a) * lvl
    return np.array(f), (lambda h: np.full(h, lvl))


def f_holt(y, a, b):
    lvl, tr, f = y[0], y[1] - y[0], [np.nan]
    for t in range(1, len(y)):
        p = lvl + tr
        f.append(p)
        new = a * y[t] + (1 - a) * p
        tr = b * (new - lvl) + (1 - b) * tr
        lvl = new
    return np.array(f), (lambda h: np.array([lvl + (i + 1) * tr for i in range(h)]))


def prepare_series(sales, interp):
    """Kembalikan (y, error, warning). y = None jika data belum layak dipakai."""
    errs, warns = [], []
    raw = pd.to_numeric(sales["permintaan"], errors="coerce")
    if raw.isna().any():
        if interp:
            raw = raw.interpolate(limit_direction="both")
            warns.append("Nilai kosong diisi dengan interpolasi linear. Pastikan ini wajar untuk usaha Anda.")
        else:
            errs.append(f"Ada {int(raw.isna().sum())} bulan dengan permintaan kosong atau bukan angka. "
                        "Lengkapi datanya, atau aktifkan opsi interpolasi di halaman Data.")
    if (raw.dropna() < 0).any():
        errs.append("Ada permintaan bernilai negatif.")
    if len(raw) < 8:
        errs.append("Data minimal 8 bulan agar tersedia data latih dan data uji.")
    elif len(raw) < 12:
        warns.append("Data kurang dari 12 bulan. Hasil peramalan dan uji statistik kurang andal.")
    if errs:
        return None, errs, warns
    return raw.values.astype(float), errs, warns


def validate_bom_par(bom, par):
    errs, warns = [], []
    if len(bom) == 0:
        errs.append("Daftar bahan (BOM) kosong.")
    if (bom["per_unit"] <= 0).any():
        warns.append("Ada bahan dengan jumlah per unit nol atau negatif.")
    if (bom["harga"] < 0).any() or (bom["stok_awal"] < 0).any():
        errs.append("Harga bahan dan stok awal bahan tidak boleh negatif.")
    return errs, warns


def analyze(y, par, cfg):
    """Evaluasi tiga metode dengan hold-out, pilih yang terbaik, lalu ramal dari seluruh data."""
    n, a, a2, b2 = cfg
    H = max(1, min(int(par["holdout"]), len(y) - 5))
    n_eff = max(2, min(n, len(y) - H - 1))
    builders = {
        "Rata-rata bergerak": lambda s: f_ma(s, n_eff),
        "Pemulusan eksponensial": lambda s: f_ses(s, a),
        "Holt (tren)": lambda s: f_holt(s, a2, b2),
    }
    start = n_eff  # indeks awal jendela evaluasi satu-langkah yang sama untuk semua metode
    out = {}
    act = y[-H:]
    for nama, bld in builders.items():
        _, fn_tr = bld(y[:-H])
        pred = fn_tr(H)
        e = act - pred
        f_full, fn_full = bld(y)
        e1 = y[start:] - f_full[start:]
        out[nama] = dict(
            f=f_full, fn=fn_full, pred_test=pred,
            mad=np.abs(e).mean(), rmse=float(np.sqrt((e ** 2).mean())),
            mape=(np.abs(e) / np.where(act == 0, 1, act)).mean() * 100,
            e1=e1, mape1=(np.abs(e1) / np.where(y[start:] == 0, 1, y[start:])).mean() * 100,
        )
    best = min(out, key=lambda k: out[k]["mape"])
    sigma = float(out[best]["e1"].std(ddof=1))
    fc = np.maximum(out[best]["fn"](par["horizon"]), 0)
    return dict(y=y, out=out, best=best, fc=fc, sigma=sigma, H=H, n_eff=n_eff, start=start)


def safety_stock(sigma, service):
    return int(np.ceil(stats.norm.ppf(service / 100) * sigma))


def par_eff(par, an):
    """Parameter efektif: stok pengaman dapat diambil otomatis dari simpangan galat ramalan."""
    if par["aman_mode"].startswith("Otomatis"):
        return {**par, "stok_aman": safety_stock(an["sigma"], par["service"])}
    return par


def plan(fc, par):
    rows, stok = [], par["stok_produk"]
    yld = par["yield_"] / 100
    kap = par["jam"] * 60 / par["waktu"]
    for i, d in enumerate(fc, 1):
        prod = max(0.0, d + par["stok_aman"] - stok)
        mulai = prod / yld
        rows.append(dict(bulan=f"Bulan +{i}", ramalan=round(d), stok_awal=round(stok),
                         produksi_baik=round(prod), unit_dimulai=round(mulai),
                         kapasitas=round(kap), utilisasi_persen=round(mulai / kap * 100, 1)))
        stok = stok + prod - d
        rows[-1]["stok_akhir"] = round(stok)
    return pd.DataFrame(rows)


def bahan_table(pl, bom):
    tab = pd.DataFrame({"bahan": bom["bahan"].values, "satuan": bom["satuan"].values})
    for _, r in pl.iterrows():
        tab[r["bulan"]] = (r["unit_dimulai"] * bom["per_unit"].values).round(2)
    kol = list(pl["bulan"])
    tab["total"] = tab[kol].sum(axis=1).round(2)
    tab["stok_awal"] = bom["stok_awal"].values
    tab["perlu_dibeli"] = (tab["total"] - tab["stok_awal"]).clip(lower=0).round(2)
    return tab


def hitung(par, bom, vol):
    yld = par["yield_"] / 100
    bahan = (bom["per_unit"] * bom["harga"]).sum()
    tk = par["waktu"] / 60 * par["tarif"]
    mulai = vol / yld
    total_var = mulai * (bahan + tk + par["oh_var"])
    total = total_var + par["oh_tetap"]
    hpp = total / vol
    if par["harga_mode"] == "Manual":
        harga = float(par["harga_manual"])
    else:
        harga = hpp * (1 + par["markup"] / 100)
    vc = total_var / vol
    bep = par["oh_tetap"] / (harga - vc) if harga > vc else np.inf
    return dict(bahan=bahan, tk=tk, hpp=hpp, harga=harga, vc=vc, bep=bep,
                laba=(harga - hpp) * vol, total=total)


def build_excel(an, pl, mat, b, vol):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Metode": list(an["out"]),
                      "MAD uji": [round(v["mad"], 2) for v in an["out"].values()],
                      "MAPE uji (%)": [round(v["mape"], 2) for v in an["out"].values()]}
                     ).to_excel(xw, sheet_name="Akurasi", index=False)
        pd.DataFrame({"Bulan": pl["bulan"], "Ramalan": an["fc"].round(1)}).to_excel(xw, sheet_name="Ramalan", index=False)
        pl.to_excel(xw, sheet_name="Rencana produksi", index=False)
        mat.to_excel(xw, sheet_name="Kebutuhan bahan", index=False)
        pd.DataFrame({"Komponen": ["Volume (unit)", "HPP per unit", "Harga jual", "BEP (unit)", "Laba bulanan"],
                      "Nilai": [vol, b["hpp"], b["harga"],
                                b["bep"] if np.isfinite(b["bep"]) else "tidak tercapai", b["laba"]]}
                     ).to_excel(xw, sheet_name="Biaya", index=False)
    return buf.getvalue()


# =====================================================================
# Antarmuka
# =====================================================================
par = st.session_state.par
bom = st.session_state.bom.fillna(0)

page = st.sidebar.radio("Menu", ["Beranda", "Data", "Peramalan", "Analisis Statistik",
                                 "Perencanaan Produksi", "Biaya", "Asumsi dan Batasan"])
st.sidebar.caption("Data bawaan adalah data simulasi. Ganti dengan data usaha di halaman Data.")


def get_analysis():
    """Analisis lengkap, atau tampilkan pesan masalah data dan hentikan halaman."""
    y, errs, _ = prepare_series(st.session_state.sales, par["interp"])
    e2, _ = validate_bom_par(bom, par)
    errs = errs + e2
    if errs:
        for m in errs:
            st.error(m)
        st.info("Perbaiki data di halaman Data.")
        st.stop()
    return analyze(y, par, st.session_state.cfg)


def vol_default(an):
    return max(1, int(round(an["fc"].mean())))


if page == "Beranda":
    st.title("Dashboard Perencanaan Produksi dan Biaya UMKM")
    st.caption("Alur: data → peramalan → rencana produksi dan bahan → biaya, harga, dan BEP.")
    an = get_analysis()
    pe = par_eff(par, an)
    fc = an["fc"]
    pl = plan(fc, pe)
    vol = vol_default(an)
    b = hitung(par, bom, vol)
    mape_best = an["out"][an["best"]]["mape"]
    c = st.columns(5)
    c[0].metric("Ramalan bulan depan", f"{fc[0]:.0f} unit",
                f"{an['best']}, MAPE uji {mape_best:.1f}%", delta_color="off")
    c[1].metric("Produksi bulan +1", f"{pl.loc[0, 'produksi_baik']} unit")
    c[2].metric("HPP per unit", rp(b["hpp"]))
    c[3].metric("BEP", f"{b['bep']:.0f} unit/bulan" if np.isfinite(b["bep"]) else "tidak tercapai")
    c[4].metric("Laba bulanan (perkiraan)", rp(b["laba"]))
    st.caption(f"Biaya dan laba dihitung pada volume {vol} unit/bulan (rata-rata ramalan) dan mengasumsikan "
               "seluruh produksi terjual pada harga jual yang dipilih. Ini perkiraan, bukan jaminan.")
    y = an["y"]
    xf = list(range(len(y), len(y) + len(fc)))
    tq = stats.t.ppf(0.975, len(an["out"][an["best"]]["e1"]) - 1)
    half = tq * an["sigma"] * np.sqrt(np.arange(1, len(fc) + 1))
    fig = go.Figure()
    fig.add_scatter(y=y, mode="lines+markers", name="Aktual")
    fig.add_scatter(x=xf, y=fc, mode="lines+markers", name="Ramalan")
    fig.add_scatter(x=xf, y=np.maximum(fc + half, 0), mode="lines", name="Batas atas 95%",
                    line=dict(dash="dot"))
    fig.add_scatter(x=xf, y=np.maximum(fc - half, 0), mode="lines", name="Batas bawah 95%",
                    line=dict(dash="dot"))
    fig.update_layout(title="Permintaan aktual dan ramalan", xaxis_title="Periode", yaxis_title="Unit")
    st.plotly_chart(fig, use_container_width=True)
    if (pl["utilisasi_persen"] > 100).any():
        st.error("Kapasitas tidak cukup pada sebagian bulan. Lihat halaman Perencanaan Produksi.")

elif page == "Data":
    st.header("Data Usaha")
    st.subheader("1. Riwayat penjualan bulanan")
    up = st.file_uploader("Unggah CSV (kolom: bulan, permintaan)", type="csv")
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.get("up_sig") != sig:  # proses sekali per file, agar editan tidak tertimpa
            d = pd.read_csv(up)
            if {"bulan", "permintaan"}.issubset(d.columns):
                st.session_state.sales = d[["bulan", "permintaan"]]
                st.session_state.up_sig = sig
                st.session_state.ver += 1
                st.success("Data penjualan dimuat.")
            else:
                st.error("Kolom harus bernama bulan dan permintaan.")
    st.download_button("Unduh template CSV penjualan", SALES0.to_csv(index=False), "template_penjualan.csv")
    par["interp"] = st.checkbox("Isi nilai kosong dengan interpolasi linear", par["interp"])
    st.session_state.sales = st.data_editor(st.session_state.sales, num_rows="dynamic",
                                            use_container_width=True, key=f"ed_sales_{st.session_state.ver}")
    _, e_s, w_s = prepare_series(st.session_state.sales, par["interp"])
    for m in e_s:
        st.error(m)
    for m in w_s:
        st.warning(m)
    st.subheader("2. Daftar bahan per unit produk (BOM)")
    st.session_state.bom = st.data_editor(st.session_state.bom, num_rows="dynamic",
                                          use_container_width=True, key="ed_bom")
    e_b, w_b = validate_bom_par(st.session_state.bom.fillna(0), par)
    for m in e_b:
        st.error(m)
    for m in w_b:
        st.warning(m)
    st.subheader("3. Parameter produksi dan biaya")
    c = st.columns(3)
    par["waktu"] = c[0].number_input("Waktu baku (menit/unit)", 0.1, value=float(par["waktu"]))
    par["tarif"] = c[1].number_input("Tarif tenaga kerja (Rp/jam)", 0, value=int(par["tarif"]))
    par["yield_"] = c[2].number_input("Yield (% produk lolos)", 1, 100, int(par["yield_"]))
    par["oh_tetap"] = c[0].number_input("Overhead tetap (Rp/bulan)", 0, value=int(par["oh_tetap"]))
    par["oh_var"] = c[1].number_input("Overhead variabel (Rp/unit)", 0, value=int(par["oh_var"]))
    par["markup"] = c[2].number_input("Markup harga jual (%)", 0, value=int(par["markup"]))
    par["jam"] = c[0].number_input("Jam kerja tersedia (jam/bulan)", 1, value=int(par["jam"]))
    par["stok_produk"] = c[1].number_input("Stok awal produk jadi (unit)", 0, value=int(par["stok_produk"]))
    par["horizon"] = c[2].slider("Horizon perencanaan (bulan)", 1, 6, int(par["horizon"]))
    st.subheader("4. Stok pengaman produk jadi")
    c = st.columns(3)
    par["aman_mode"] = c[0].radio("Sumber stok pengaman", ["Manual", "Otomatis (dari galat ramalan)"],
                                  index=0 if par["aman_mode"] == "Manual" else 1)
    par["stok_aman"] = c[1].number_input("Stok pengaman manual (unit)", 0, value=int(par["stok_aman"]))
    par["service"] = c[2].selectbox("Tingkat layanan (mode otomatis)", [90, 95, 97, 99],
                                    index=[90, 95, 97, 99].index(int(par["service"])))

elif page == "Peramalan":
    st.header("Peramalan Permintaan")
    n0, a0, a20, b20 = st.session_state.cfg
    c = st.columns(5)
    n = c[0].slider("Periode rata-rata bergerak", 2, 6, n0)
    a = c[1].slider("Alpha (pemulusan)", 0.05, 0.95, a0, 0.05)
    a2 = c[2].slider("Alpha (Holt)", 0.05, 0.95, a20, 0.05)
    b2 = c[3].slider("Beta (Holt)", 0.05, 0.95, b20, 0.05)
    par["holdout"] = c[4].slider("Data uji (bulan terakhir)", 1, 4, int(par["holdout"]))
    st.session_state.cfg = (n, a, a2, b2)
    an = get_analysis()
    out, best, fc, H, y = an["out"], an["best"], an["fc"], an["H"], an["y"]
    st.caption(f"Pemilihan metode memakai validasi hold-out: model dibentuk dari data sebelum {H} bulan terakhir, "
               f"lalu diuji pada {H} bulan terakhir itu. Kolom 'satu-langkah' adalah galat ramalan bulan demi bulan "
               "pada jendela data yang sama untuk semua metode.")
    tab = pd.DataFrame([{"Metode": k, "MAD uji": round(v["mad"], 2), "RMSE uji": round(v["rmse"], 2),
                         "MAPE uji (%)": round(v["mape"], 2),
                         "MAPE satu-langkah (%)": round(v["mape1"], 2),
                         "Ramalan bulan +1": round(float(v["fn"](1)[0]), 1)} for k, v in out.items()])
    st.dataframe(tab, hide_index=True, use_container_width=True)
    st.success(f"Metode terpilih: {best} (MAPE data uji terkecil). Ramalan {par['horizon']} bulan: "
               + ", ".join(f"{x:.0f}" for x in fc))
    if H < 3:
        st.warning("Data uji hanya 1-2 bulan, pemilihan metode sangat dipengaruhi kebetulan. Tambah data bila bisa.")
    fig = go.Figure()
    fig.add_scatter(y=y, mode="lines+markers", name="Aktual")
    for k, v in out.items():
        fig.add_scatter(y=v["f"], mode="lines", name=k)
    fig.add_scatter(x=list(range(len(y), len(y) + len(fc))), y=fc, mode="lines+markers",
                    name=f"Ramalan ({best})", line=dict(dash="dash"))
    fig.add_vrect(x0=len(y) - H - 0.5, x1=len(y) - 0.5, fillcolor="gray", opacity=0.15,
                  annotation_text="data uji", annotation_position="top left")
    fig.update_layout(xaxis_title="Periode", yaxis_title="Unit")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Analisis Statistik":
    st.header("Analisis Statistik")
    an = get_analysis()
    out, best, y = an["out"], an["best"], an["y"]
    e1 = out[best]["e1"]
    st.caption(f"Dasar: galat ramalan satu-langkah metode {best} pada {len(e1)} bulan. "
               "Dengan data sebanyak ini uji statistik berkekuatan terbatas, jadi baca hasilnya sebagai indikasi.")

    st.subheader("1. Uji tren permintaan")
    t_idx = np.arange(len(y))
    lr = stats.linregress(t_idx, y)
    c = st.columns(3)
    c[0].metric("Kemiringan (unit/bulan)", f"{lr.slope:.2f}")
    c[1].metric("p-value", f"{lr.pvalue:.4f}")
    c[2].metric("R²", f"{lr.rvalue ** 2:.2f}")
    if lr.pvalue < 0.05:
        st.write("Tren linear signifikan pada α = 5%, sehingga metode yang memuat tren (Holt) beralasan dipakai.")
    else:
        st.write("Tren linear tidak signifikan pada α = 5%. Metode tanpa tren patut dipertimbangkan.")

    st.subheader("2. Uji kenormalan galat (Shapiro-Wilk)")
    if len(e1) >= 3:
        sw = stats.shapiro(e1)
        c = st.columns(2)
        c[0].metric("Statistik W", f"{sw.statistic:.3f}")
        c[1].metric("p-value", f"{sw.pvalue:.4f}")
        if sw.pvalue >= 0.05:
            st.write("Galat tidak menyimpang nyata dari normal, sehingga interval prediksi dan stok pengaman "
                     "berbasis distribusi normal dapat dipertanggungjawabkan.")
        else:
            st.warning("Galat menyimpang dari normal. Interval dan stok pengaman di bawah bersifat kurang akurat.")
    else:
        st.info("Galat terlalu sedikit untuk uji kenormalan.")

    st.subheader("3. Uji t berpasangan antar dua metode")
    c = st.columns(2)
    m1 = c[0].selectbox("Metode A", list(out), index=0)
    m2 = c[1].selectbox("Metode B", list(out), index=2)
    if m1 == m2:
        st.info("Pilih dua metode yang berbeda.")
    else:
        ea, eb = np.abs(out[m1]["e1"]), np.abs(out[m2]["e1"])
        tt = stats.ttest_rel(ea, eb)
        c = st.columns(3)
        c[0].metric(f"Rata-rata |galat| {m1}", f"{ea.mean():.2f}")
        c[1].metric(f"Rata-rata |galat| {m2}", f"{eb.mean():.2f}")
        c[2].metric("p-value", f"{tt.pvalue:.4f}")
        if tt.pvalue < 0.05:
            lebih = m1 if ea.mean() < eb.mean() else m2
            st.write(f"Selisih akurasi signifikan pada α = 5%; {lebih} lebih akurat.")
        else:
            st.write("Selisih akurasi tidak signifikan pada α = 5%. Kedua metode belum dapat dibedakan "
                     "berdasarkan data yang ada.")

    st.subheader("4. Interval prediksi 95%")
    tq = stats.t.ppf(0.975, len(e1) - 1)
    half = tq * an["sigma"] * np.sqrt(np.arange(1, len(an["fc"]) + 1))
    ip = pd.DataFrame({"Bulan": [f"Bulan +{i}" for i in range(1, len(an["fc"]) + 1)],
                       "Ramalan": an["fc"].round(1),
                       "Batas bawah": np.maximum(an["fc"] - half, 0).round(1),
                       "Batas atas": (an["fc"] + half).round(1)})
    st.dataframe(ip, hide_index=True, use_container_width=True)
    st.caption("Lebar interval = t × σ × √h, dengan σ simpangan baku galat satu-langkah. "
               "Ini pendekatan sederhana; ketidakpastian parameter tidak dihitung.")

    st.subheader("5. Stok pengaman dari galat ramalan")
    z = stats.norm.ppf(par["service"] / 100)
    ss = safety_stock(an["sigma"], par["service"])
    c = st.columns(4)
    c[0].metric("σ galat (unit)", f"{an['sigma']:.1f}")
    c[1].metric("Tingkat layanan", f"{par['service']}%")
    c[2].metric("z", f"{z:.2f}")
    c[3].metric("Stok pengaman = z × σ", f"{ss} unit")
    if par["aman_mode"].startswith("Otomatis"):
        st.success("Mode otomatis aktif. Nilai ini dipakai di Perencanaan Produksi.")
    else:
        st.info(f"Mode manual aktif ({par['stok_aman']} unit). Ubah sumber stok pengaman di halaman Data "
                "untuk memakai nilai ini.")

elif page == "Perencanaan Produksi":
    st.header("Perencanaan Produksi dan Kebutuhan Bahan")
    an = get_analysis()
    pe = par_eff(par, an)
    pl = plan(an["fc"], pe)
    st.caption(f"Dasar: ramalan metode {an['best']}. Produksi = ramalan + stok pengaman ({pe['stok_aman']} unit, "
               f"{pe['aman_mode'].lower()}) − stok awal, unit dimulai = produksi ÷ yield, "
               "kapasitas = jam tersedia × 60 ÷ waktu baku.")
    st.dataframe(pl, hide_index=True, use_container_width=True)
    fig = go.Figure()
    fig.add_bar(x=pl["bulan"], y=pl["unit_dimulai"], name="Unit dimulai")
    fig.add_scatter(x=pl["bulan"], y=pl["kapasitas"], name="Kapasitas", mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)
    if (pl["utilisasi_persen"] > 100).any():
        st.error("Ada bulan dengan utilisasi di atas 100 persen. Tambah jam kerja, tenaga, atau geser produksi.")
    st.subheader("Kebutuhan bahan (MRP sederhana)")
    mat = bahan_table(pl, bom)
    st.dataframe(mat, hide_index=True, use_container_width=True)
    c = st.columns(2)
    c[0].download_button("Unduh rencana (CSV)", pl.to_csv(index=False), "rencana_produksi.csv")
    b = hitung(par, bom, vol_default(an))
    c[1].download_button("Unduh ringkasan (Excel)", build_excel(an, pl, mat, b, vol_default(an)),
                         "ringkasan_perencanaan.xlsx")

elif page == "Biaya":
    st.header("Harga Pokok Produksi, Harga Jual, dan BEP")
    an = get_analysis()
    vol = st.number_input("Volume produksi baik per bulan (unit)", 1, value=vol_default(an))
    par["harga_mode"] = st.radio("Penentuan harga jual", ["Markup dari HPP", "Manual"], horizontal=True,
                                 index=0 if par["harga_mode"] != "Manual" else 1)
    if par["harga_mode"] == "Manual":
        par["harga_manual"] = st.number_input("Harga jual per unit (Rp)", 1, value=int(par["harga_manual"]))
    else:
        st.caption("Pada mode markup, harga jual ikut naik saat HPP naik, sehingga laba selalu positif. "
                   "Gunakan mode manual untuk menguji harga pasar sebenarnya.")
    b = hitung(par, bom, vol)
    c = st.columns(4)
    c[0].metric("HPP per unit", rp(b["hpp"]))
    c[1].metric("Harga jual", rp(b["harga"]))
    c[2].metric("BEP", f"{b['bep']:.0f} unit" if np.isfinite(b["bep"]) else "tidak tercapai")
    c[3].metric("Laba bulanan", rp(b["laba"]))
    if b["harga"] < b["hpp"]:
        st.warning("Harga jual di bawah HPP pada volume ini, usaha merugi per unit.")
    rinci = pd.DataFrame({
        "Komponen": ["Bahan per unit dimulai", "Tenaga kerja per unit", "Overhead variabel per unit",
                     "Overhead tetap per bulan", "Total biaya produksi per bulan"],
        "Nilai": [rp(b["bahan"]), rp(b["tk"]), rp(par["oh_var"]), rp(par["oh_tetap"]), rp(b["total"])]})
    st.dataframe(rinci, hide_index=True, use_container_width=True)
    st.subheader("Analisis sensitivitas (perubahan HPP)")
    rows = []
    for nama, fn in {
        "Harga bahan +10%": lambda p, bm, v: (p, bm.assign(harga=bm["harga"] * 1.1), v),
        "Waktu baku +10%": lambda p, bm, v: ({**p, "waktu": p["waktu"] * 1.1}, bm, v),
        "Yield −10%": lambda p, bm, v: ({**p, "yield_": p["yield_"] * 0.9}, bm, v),
        "Volume −10%": lambda p, bm, v: (p, bm, v * 0.9),
    }.items():
        p2, bm2, v2 = fn(par, bom, vol)
        r2 = hitung(p2, bm2, v2)
        rows.append({"Skenario": nama, "HPP baru": rp(r2["hpp"]),
                     "Perubahan HPP (%)": round((r2["hpp"] / b["hpp"] - 1) * 100, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

elif page == "Asumsi dan Batasan":
    st.header("Asumsi dan Batasan")
    st.markdown("""
**Tujuan.** Membantu UMKM memperkirakan permintaan, menyusun rencana produksi dan kebutuhan bahan, serta
mengetahui HPP, harga jual, dan titik impas.

**Metode.** Peramalan memakai rata-rata bergerak, pemulusan eksponensial, dan Holt. Metode dipilih dari
MAPE pada data uji (hold-out). Interval prediksi dan stok pengaman memakai simpangan baku galat satu-langkah.
Biaya memakai full costing, BEP = overhead tetap ÷ (harga − biaya variabel per unit).

**Asumsi.**
- Pola permintaan masa depan mengikuti pola historis.
- Usaha membuat satu jenis produk dengan satu komposisi bahan.
- Galat ramalan berdistribusi mendekati normal dan tidak saling berkaitan (diperiksa di halaman Analisis Statistik).
- Seluruh produksi baik terjual pada harga yang ditetapkan.
- Waktu baku, yield, tarif, dan overhead tetap konstan selama horizon perencanaan.

**Batasan.**
- Data bawaan adalah simulasi, belum mewakili usaha tertentu.
- Dengan 12 bulan data, pola musiman tidak dapat diestimasi dan uji statistik berkekuatan rendah.
- Parameter pemulusan diatur manual lewat slider, tidak dioptimasi otomatis.
- Belum ada diskon kuantitas, lead time pemasok, atau kendala modal kerja.
- Kapasitas hanya mempertimbangkan jam kerja, bukan mesin atau tenaga kerja secara terpisah.
""")
