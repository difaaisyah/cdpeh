import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
PAR0 = dict(tarif=25000, waktu=20.0, oh_tetap=1500000, oh_var=500, yield_=95,
            markup=40, jam=160, stok_produk=30, stok_aman=20, horizon=3)

st.session_state.setdefault("sales", SALES0.copy())
st.session_state.setdefault("bom", BOM0.copy())
st.session_state.setdefault("par", dict(PAR0))
st.session_state.setdefault("cfg", (3, 0.3, 0.5, 0.2))  # n, alpha, alpha Holt, beta Holt


def rp(x):
    return "Rp{:,.0f}".format(x).replace(",", ".")


# ---------- Peramalan ----------
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


def series():
    d = pd.to_numeric(st.session_state.sales["permintaan"], errors="coerce").dropna()
    return d.values.astype(float)


def run_all(y, n, a, a2, b2):
    out = {}
    metode = {"Rata-rata bergerak": f_ma(y, n), "Pemulusan eksponensial": f_ses(y, a),
              "Holt (tren)": f_holt(y, a2, b2)}
    for nama, (f, fn) in metode.items():
        m = ~np.isnan(f)
        e = np.abs(y[m] - f[m])
        out[nama] = dict(f=f, fn=fn, mad=e.mean(), mape=(e / np.where(y[m] == 0, 1, y[m])).mean() * 100)
    best = min(out, key=lambda k: out[k]["mape"])
    return out, best


def forecast(h):
    y = series()
    if len(y) < 4:
        return None
    n, a, a2, b2 = st.session_state.cfg
    out, best = run_all(y, min(n, len(y) - 1), a, a2, b2)
    return out, best, np.maximum(out[best]["fn"](h), 0)


# ---------- Perencanaan produksi ----------
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


# ---------- Biaya (full costing) ----------
def hitung(par, bom, vol):
    yld = par["yield_"] / 100
    bahan = (bom["per_unit"] * bom["harga"]).sum()
    tk = par["waktu"] / 60 * par["tarif"]
    mulai = vol / yld
    total_var = mulai * (bahan + tk + par["oh_var"])
    total = total_var + par["oh_tetap"]
    hpp = total / vol
    harga = hpp * (1 + par["markup"] / 100)
    vc = total_var / vol
    bep = par["oh_tetap"] / (harga - vc) if harga > vc else np.inf
    return dict(bahan=bahan, tk=tk, hpp=hpp, harga=harga, vc=vc, bep=bep,
                laba=(harga - hpp) * vol, total=total)


def vol_default():
    r = forecast(st.session_state.par["horizon"])
    return max(1, int(round(r[2].mean()))) if r else 100


# ---------- Navigasi ----------
page = st.sidebar.radio("Menu", ["Beranda", "Data", "Peramalan", "Perencanaan Produksi", "Biaya"])
st.sidebar.caption("Data bawaan adalah data simulasi. Ganti dengan data usaha di halaman Data.")
par = st.session_state.par
bom = st.session_state.bom.fillna(0)

if page == "Beranda":
    st.title("Dashboard Perencanaan Produksi dan Biaya UMKM")
    st.caption("Alur: data → peramalan → rencana produksi dan bahan → biaya, harga, dan BEP.")
    res = forecast(par["horizon"])
    if res is None:
        st.warning("Data penjualan minimal 4 bulan. Lengkapi di halaman Data.")
    else:
        out, best, fc = res
        pl = plan(fc, par)
        vol = vol_default()
        b = hitung(par, bom, vol)
        c = st.columns(5)
        c[0].metric("Ramalan bulan depan", f"{fc[0]:.0f} unit", best)
        c[1].metric("Produksi bulan +1", f"{pl.loc[0, 'produksi_baik']} unit")
        c[2].metric("HPP per unit", rp(b["hpp"]))
        c[3].metric("BEP", f"{b['bep']:.0f} unit/bulan" if np.isfinite(b["bep"]) else "tidak tercapai")
        c[4].metric("Laba bulanan (perkiraan)", rp(b["laba"]))
        y = series()
        fig = go.Figure()
        fig.add_scatter(y=y, mode="lines+markers", name="Aktual")
        fig.add_scatter(x=list(range(len(y), len(y) + len(fc))), y=fc, mode="lines+markers", name="Ramalan")
        fig.update_layout(title="Permintaan aktual dan ramalan", xaxis_title="Periode", yaxis_title="Unit")
        st.plotly_chart(fig, use_container_width=True)
        if (pl["utilisasi_persen"] > 100).any():
            st.error("Kapasitas tidak cukup pada sebagian bulan. Lihat halaman Perencanaan Produksi.")

elif page == "Data":
    st.header("Data Usaha")
    st.subheader("1. Riwayat penjualan bulanan")
    up = st.file_uploader("Unggah CSV (kolom: bulan, permintaan)", type="csv")
    if up is not None:
        d = pd.read_csv(up)
        if {"bulan", "permintaan"}.issubset(d.columns):
            st.session_state.sales = d[["bulan", "permintaan"]]
            st.success("Data penjualan dimuat.")
        else:
            st.error("Kolom harus bernama bulan dan permintaan.")
    st.download_button("Unduh template CSV penjualan", SALES0.to_csv(index=False), "template_penjualan.csv")
    st.session_state.sales = st.data_editor(st.session_state.sales, num_rows="dynamic",
                                            use_container_width=True, key="ed_sales")
    st.subheader("2. Daftar bahan per unit produk (BOM)")
    st.session_state.bom = st.data_editor(st.session_state.bom, num_rows="dynamic",
                                          use_container_width=True, key="ed_bom")
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
    par["stok_aman"] = c[2].number_input("Stok pengaman produk (unit)", 0, value=int(par["stok_aman"]))
    par["horizon"] = c[0].slider("Horizon perencanaan (bulan)", 1, 6, int(par["horizon"]))

elif page == "Peramalan":
    st.header("Peramalan Permintaan")
    n0, a0, a20, b20 = st.session_state.cfg
    c = st.columns(4)
    n = c[0].slider("Periode rata-rata bergerak", 2, 6, n0)
    a = c[1].slider("Alpha (pemulusan)", 0.05, 0.95, a0, 0.05)
    a2 = c[2].slider("Alpha (Holt)", 0.05, 0.95, a20, 0.05)
    b2 = c[3].slider("Beta (Holt)", 0.05, 0.95, b20, 0.05)
    st.session_state.cfg = (n, a, a2, b2)
    res = forecast(par["horizon"])
    if res is None:
        st.warning("Data penjualan minimal 4 bulan.")
    else:
        out, best, fc = res
        tab = pd.DataFrame([{"Metode": k, "MAD": round(v["mad"], 2), "MAPE (%)": round(v["mape"], 2),
                             "Ramalan bulan +1": round(float(v["fn"](1)[0]), 1)} for k, v in out.items()])
        st.dataframe(tab, hide_index=True, use_container_width=True)
        st.success(f"Metode terpilih: {best} (MAPE terkecil). Ramalan {par['horizon']} bulan: "
                   + ", ".join(f"{x:.0f}" for x in fc))
        y = series()
        fig = go.Figure()
        fig.add_scatter(y=y, mode="lines+markers", name="Aktual")
        for k, v in out.items():
            fig.add_scatter(y=v["f"], mode="lines", name=k)
        fig.update_layout(xaxis_title="Periode", yaxis_title="Unit")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Perencanaan Produksi":
    st.header("Perencanaan Produksi dan Kebutuhan Bahan")
    res = forecast(par["horizon"])
    if res is None:
        st.warning("Data penjualan minimal 4 bulan.")
    else:
        out, best, fc = res
        pl = plan(fc, par)
        st.caption(f"Dasar: ramalan metode {best}. Produksi = ramalan + stok pengaman − stok awal, "
                   "unit dimulai = produksi ÷ yield, kapasitas = jam tersedia × 60 ÷ waktu baku.")
        st.dataframe(pl, hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(x=pl["bulan"], y=pl["unit_dimulai"], name="Unit dimulai")
        fig.add_scatter(x=pl["bulan"], y=pl["kapasitas"], name="Kapasitas", mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
        if (pl["utilisasi_persen"] > 100).any():
            st.error("Ada bulan dengan utilisasi di atas 100 persen. Tambah jam kerja, tenaga, atau geser produksi.")
        st.subheader("Kebutuhan bahan (MRP sederhana)")
        tab = pd.DataFrame({"bahan": bom["bahan"], "satuan": bom["satuan"]})
        for _, r in pl.iterrows():
            tab[r["bulan"]] = (r["unit_dimulai"] * bom["per_unit"]).round(2)
        kol = list(pl["bulan"])
        tab["total"] = tab[kol].sum(axis=1).round(2)
        tab["stok_awal"] = bom["stok_awal"]
        tab["perlu_dibeli"] = (tab["total"] - tab["stok_awal"]).clip(lower=0).round(2)
        st.dataframe(tab, hide_index=True, use_container_width=True)
        st.download_button("Unduh rencana (CSV)", pl.to_csv(index=False), "rencana_produksi.csv")

elif page == "Biaya":
    st.header("Harga Pokok Produksi, Harga Jual, dan BEP")
    vol = st.number_input("Volume produksi baik per bulan (unit)", 1, value=vol_default())
    b = hitung(par, bom, vol)
    c = st.columns(4)
    c[0].metric("HPP per unit", rp(b["hpp"]))
    c[1].metric("Harga jual", rp(b["harga"]))
    c[2].metric("BEP", f"{b['bep']:.0f} unit" if np.isfinite(b["bep"]) else "tidak tercapai")
    c[3].metric("Laba bulanan", rp(b["laba"]))
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
        h2 = hitung(p2, bm2, v2)["hpp"]
        rows.append({"Skenario": nama, "HPP baru": rp(h2), "Perubahan (%)": round((h2 / b["hpp"] - 1) * 100, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
