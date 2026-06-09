# Battery-Management-Thermal-System

## Model Predictive Control (MPC) untuk Sistem Manajemen Termal Baterai EV

Proyek Akhir — Sistem Kendali Prediktif dan Adaptif 2026

**Kelompok 2**

| Nama | NPM |
|------|-----|
| Fathih Rayyandra Firmansyah | 2306205260 |
| Muhammad Alif Iqbal | 2306206654 |
| Rafsya Ghaizan Athari | 2306210052 |

Departemen Teknik Elektro — Universitas Indonesia

---

Studi komparatif **8 varian Model Predictive Control (MPC)** untuk manajemen termal baterai kendaraan listrik (EV).

---

## Deskripsi

Proyek ini mensimulasikan **model termal 2-state** baterai EV (temperatur inti dan temperatur permukaan) dengan:

- **1 input kontrol**: laju aliran massa coolant (kg/s)
- **1 gangguan terukur**: pembangkitan panas dari drive-cycle (Watt)
- **1 output**: temperatur inti baterai, **setpoint 30 °C**
- **Durasi simulasi**: 2000 detik, sampling 1 detik

8 varian MPC disimulasikan pada plant termal yang sama dan dibandingkan performanya.

---

## 8 Varian MPC

| # | Varian | Deskripsi |
|---|--------|-----------|
| 1 | **Basic MPC** | QP-based MPC baseline dengan biaya tracking (`quadprog`) |
| 2 | **Robust MPC** | Constraint tightening dengan model plant worst-case (±20% R_cond, ±10% C_core) |
| 3 | **Stochastic MPC** | Chance-constrained: P(T_core ≤ 35 °C) ≥ 0.95 (κ = 1.645) |
| 4 | **Economic MPC** | Biaya energi pompa + penalti over-temperature (α = 2.0, β = 50) |
| 5 | **Soft Constraints MPC** | Slack variables dengan penalti besar (ρ = 1000) pada pelanggaran constraint |
| 6 | **Explicit MPC** | Lookup table 3D berbasis grid (9×9×20, interpolasi trilinear) |
| 7 | **Distributed MPC** | Konsensus dual-agent (2 iterasi, weighted average) |
| 8 | **Kalman + MPC** | Output feedback via Kalman filter estimator dengan pengukuran noise |

---

## Parameter Model Termal

| Parameter | Nilai | Satuan |
|-----------|-------|--------|
| C_core | 10,000 | J/°C |
| C_surface | 3,000 | J/°C |
| R_cond | 0.005 | °C/W |
| cp | 4,200 | J/(kg·°C) |
| T_cool_in | 25 | °C |

### Parameter MPC

| Parameter | Nilai |
|-----------|-------|
| Ts | 1 s |
| Durasi | 2000 s |
| Np (prediction horizon) | 10 |
| Nc (control horizon) | 3 |
| Qw (tracking weight) | 10 |
| Rw (control move weight) | 0.1 |
| u_min, u_max | 0, 5 kg/s |
| Setpoint | 30 °C |
| T_init | 20 °C |

---

## Struktur Proyek

```
Battery-Management-Thermal-System/
├── main.m                  # Skrip utama (706 baris)
├── data/                   # Direktori data (kosong)
├── images/                 # Output plot
│   ├── Battery Core Temperature.png
│   ├── Control Action.png
│   ├── Drive-Cycle_Heat_Disturbance.png
│   └── Kalman VS True Core Temp.png
├── program/                # Duplikat main.m
│   └── main.m
├── presentation/           # Slide presentasi kelompok
│   └── Presentasi_Kelompok-2.pdf
├── references/             # Paper referensi (5 PDF)
├── reports/                # Laporan kelompok
│   └── Reports_Kelompok-2.pdf
└── README.md
```

---

## Dependensi MATLAB

- MATLAB R2016b atau lebih baru
- **Control System Toolbox** (`ss`, `c2d`)
- **Optimization Toolbox** (`quadprog`)

---

## Cara Menjalankan

1. Buka MATLAB dan arahkan ke direktori proyek.
2. Jalankan skrip utama:

```matlab
main
```

atau

```matlab
run('main.m')
```

### Output yang Dihasilkan

- **Figure 1**: Perbandingan tracking temperatur (8 varian MPC + no-control baseline)
- **Figure 2**: Profil disturbance + estimasi Kalman vs temperatur sebenarnya
- **Console table**: Perbandingan performa (RMSE, avg flow, konsumsi energi)
- Estimasi waktu eksekusi: ~2–5 menit

---

## Temuan Utama

- **Robust MPC** mengurangi laju aliran ~4% dengan penalti RMSE minimal
- **Economic MPC** menghasilkan perilaku deadband on/off
- **Stochastic / Soft / Distributed MPC** berperilaku identik dengan Basic MPC pada plant yang well-behaved ini
- **Kalman + MPC** mencapai rasio energi/performa terbaik (menghemat ~92% energi coolant dengan selisih RMSE hanya 0.47 °C)

---

## Referensi

1. Recent journal paper on MPC for battery thermal management (2025+)
2. *Adaptive Model Predictive Control for Battery Thermal Management for Electric Vehicles*
3. *Distributed Model Predictive Controller for Thermal Energy Management System of Battery Electric Vehicles*
4. *MPC-based Battery Thermal Management Controller for Plug-in Hybrid Electric Vehicles*
5. *Optimization of Battery Thermal Management in Electric Vehicles: a Model Predictive Control Approach Integrated with Particle Swarm Optimization*
