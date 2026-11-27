# Primary Actuarial Source Documents & Traceability

`pricing-kernels` implements published, peer-reviewed property & casualty commercial lines actuarial pricing, classification, experience rating, retrospective plans, credibility, severity layering, and trending methods directly from primary literature. Every kernel is numerically verified in [`tests/source_benchmarks.rs`](../tests/source_benchmarks.rs).

---

## 1. Classification Ratemaking & Relativity Estimation

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **Bailey & Simon (1960)** | R. A. Bailey & L. J. Simon, *"Two Studies in Automobile Insurance"*, *Proceedings of the Casualty Actuarial Society (PCAS)*, Vol. XLVII (1960), pp. 1–19. | `src/classification/minimum_bias.rs` (`bailey_simon_chi_squared`) — Weighted $\chi^2$ minimum bias coordinate updates $r_i = \sqrt{\sum_j P_{ij} A_{ij}^2 / c_j \big/ \sum_j P_{ij} c_j}$. | Strict $\chi^2$ optimality over balance-principle fit and exact multiplicative recovery. |
| **Bailey (1963) / Brown (1988) / CAS Monograph 5** | R. A. Bailey, *"Insurance Rates with Minimum Bias"*, *PCAS*, Vol. L (1963), pp. 4–11; R. L. Brown, *"Minimum Bias with Generalized Linear Models"*, *PCAS*, Vol. LXXV (1988), pp. 187–216; Goldburd, Khare, Tevet, *Generalized Linear Models for Insurance Rating*, CAS Monograph No. 5. | `src/classification/minimum_bias.rs` (`bailey_simon_multiplicative`), `src/classification/irls_glm.rs` (`fit_glm_irls`) — Balance-principle equations $\sum_j P_{ij}(A_{ij} - r_i c_j) = 0$; weighted IRLS for Poisson, Quasi-Poisson, and Gamma log-link GLMs with exposure offsets, exact post-convergence Fisher information, and Pearson $\chi^2 / \text{df}$ dispersion. | Exact row/column balance to `1e-9`; exact equivalence between Bailey (1963) balance relativities and Poisson log-link GLM with $\ln(P_{ij})$ offsets (Brown 1988); weighted Gamma MLE parity; singular matrix rejection. |
| **Werner & Modlin (CAS Exam 5)** | G. Werner & C. Modlin, *Basic Ratemaking*, CAS (2016), Ch. 7–8. | `src/classification/scoring.rs` — Zero-allocation multiplicative rating plan evaluator $\text{Base} \times \text{Exposure} \times \prod_k r_k$ with minimum and maximum premium bounds. | Single and batch evaluation bounds and slice-mismatch guards. |

---

## 2. Credibility Theory

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **Longley-Cook (1962)** | L. H. Longley-Cook, *"An Introduction to Credibility Theory"*, *PCAS*, Vol. XLIX (1962), pp. 194–221. | `src/credibility/classical.rs` — Classical/limited fluctuation credibility $Z = \sqrt{\min(1, N / N_{\text{full}})}$; standard 90/5 Poisson benchmark ($N_{\text{full}} = 1082.41$). | Longley-Cook standard benchmarks and partial credibility blending. |
| **Bühlmann & Straub (1970)** | H. Bühlmann & E. Straub, *"Glaubwürdigkeit für Schadensätze"*, *Bulletin of the Swiss Association of Actuaries* (1970), pp. 111–133. | `src/credibility/buhlmann_straub.rs` — Non-parametric Bühlmann-Straub credibility for non-uniform exposures: Expected Process Variance ($s^2$), Variance of Hypothetical Means ($a$), $K = s^2 / a$, and group weights $Z_i = P_i / (P_i + K)$. | Hand-verified exact analytical EPV, VHM, $K$, and $Z_i$ to `1e-12`. |

---

## 3. Commercial Experience & Retrospective Rating

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **NCCI Experience Rating Plan** | NCCI, *Experience Rating Plan Manual for Workers Compensation and Employers Liability Insurance*, Rule 2. | `src/experience/ncci_erp.rs` — Exact NCCI ERP formula $\text{Mod} = (A_p + W A_e + (1 - W) E_e + B) / (E + B)$; split point truncation; ERA 70% medical-only reduction; off-balance adjustment factor $B = \sum E / \sum (E \cdot M)$. | Exact NCCI worksheet calculation (`107,225 / 125,000 = 0.8578 -> 0.858`) and off-balance identity. |
| **Lee (1988)** | Y.-S. Lee, *"The Mathematics of Excess of Loss Coverages and Retrospective Rating — A Graphical Approach"*, *PCAS*, Vol. LXXV (1988), pp. 49–77. | `src/experience/table_m.rs` — Insurance charge $\phi(r) = E[\max(0, Y - r)]$ and savings $\psi(r) = \phi(r) + r - 1$; exact $\phi'(r) = -1$ below minimum support point; automatic sorting and duplicate merging. | Exact Table M evaluations below and above support points; unsorted input invariance. |
| **Brosius (1989) / CAS Exam 8** | E. Brosius, *"Table M Construction"*, CAS Study Note (1989); CAS Exam 8 Retrospective Rating syllabus. | `src/experience/retrospective.rs` — Actuarially balanced Retrospective Rating Plan solver: Basic Premium Factor ($\text{BPF}$) iteration under Maximum ($G$) and Minimum ($H$) constraints with post-convergence consistency refresh and infeasibility detection. | Exact actuarial balance equation verification to `1e-9`. |

---

## 4. Severity Layering, ILFs, Deductibles, and Trending

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **Klugman, Panjer & Willmot / Miccolis (1977) / Lee (1988) / Keatinge (1999)** | S. A. Klugman, H. H. Panjer & G. E. Willmot, *Loss Models: From Data to Decisions* (5th ed.), Appendix A; R. S. Miccolis, *PCAS* LXIV (1977); Y.-S. Lee, *PCAS* LXXV (1988); C. L. Keatinge, *PCAS* LXXXVI (1999). | `src/layers/ilf.rs` — Exact closed-form LEV $E[X \wedge L]$, survival $S(x)$, unbounded mean, Increased Limit Factors $\text{ILF}(L; B)$, Loss Elimination Ratios $\text{LER}(d; L)$, excess ratios, and `CommercialLayer` pricing across all 6 parametric families: Lognormal (Cody 1969 `erfc`), Lomax (Pareto Type II), Single-Parameter Pareto, Exponential, ISO Mixed Exponential, and GPD. | Numerical survival-integral verification $\int_0^L S(x)\,dx = E[X \wedge L]$, layer sum conservation, and excess-ratio conservation across all 6 families. |
| **Kallop (1975) / Werner & Modlin (CAS Exam 5)** | R. Kallop, *"A Current Look at Workers Compensation Ratemaking"*, *PCAS*, Vol. LXII (1975); Werner & Modlin (2016), Ch. 5–6. | `src/trending/parallelogram.rs`, `src/trending/trend.rs` — Exact telescoping-area Calendar-Year earned parallelogram (`parallelogram_on_level_factor`) and Policy-Year written (`policy_year_on_level_factor`) on-leveling for arbitrarily many unsorted rate changes; exponential trend model fitting via log-linear regression. | Multi-change unsorted Calendar-Year (including prior-year earning) and Policy-Year exact geometric area verification to `1e-12`. |
