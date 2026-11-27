# Primary Actuarial Source Documents & Traceability

`pricing-kernels` implements published, peer-reviewed property & casualty commercial lines actuarial pricing, classification, experience rating, retrospective plans, credibility, severity layering, and trending methods directly from primary literature. Every kernel is numerically verified in [`tests/source_benchmarks.rs`](../tests/source_benchmarks.rs).

---

## 1. Classification Ratemaking & Relativity Estimation

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **Bailey & Simon (1960)** | R. A. Bailey & L. J. Simon, *"Two Studies in Automobile Insurance"*, *Proceedings of the Casualty Actuarial Society (PCAS)*, Vol. XLVII (1960), pp. 1–19. | `src/classification/minimum_bias.rs` — Multiplicative minimum bias balance equations $\sum_j P_{ij}(A_{ij} - r_i c_j) = 0$ and $\sum_i P_{ij}(A_{ij} - r_i c_j) = 0$; $\chi^2$ goodness-of-fit diagnostic. | Bailey & Simon (1960) Canadian automobile merit rating data (Table 1); exact reproduction of synthetic 2×2 balance. |
| **Brown (1988) / CAS Monograph 5** | R. L. Brown, *"Minimum Bias with Generalized Linear Models"*, *PCAS*, Vol. LXXV (1988), pp. 187–216; Goldburd, Khare, Tevet, *Generalized Linear Models for Insurance Rating*, CAS Monograph No. 5. | `src/classification/irls_glm.rs` — Weighted IRLS algorithm for Poisson (log link, exposure offset) and Gamma (log link, frequency weights) families; deviance and asymptotic parameter standard errors. | Poisson count regression & Gamma loss severity parameter convergence. |
| **Casualty Actuarial Society Exam 5** | Werner & Modlin, *Basic Ratemaking*, CAS (2016), Ch. 7–8. | `src/classification/scoring.rs` — Zero-allocation multiplicative rating plan evaluator $\text{Base} \times \prod_k r_k$ with minimum and maximum premium bounds. | Unit tests for single and batch evaluation bounds. |

---

## 2. Credibility Theory

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **Longley-Cook (1962)** | L. H. Longley-Cook, *"An Introduction to Credibility Theory"*, *PCAS*, Vol. XLIX (1962), pp. 194–221. | `src/credibility/classical.rs` — Classical/limited fluctuation credibility $Z = \sqrt{\min(1, N / N_{\text{full}})}$; standard 90/5 Poisson benchmark ($N_{\text{full}} = 1082.41$). | Longley-Cook standard benchmarks and partial credibility blending. |
| **Bühlmann & Straub (1970)** | H. Bühlmann & E. Straub, *"Glaubwürdigkeit für Schadensätze"*, *Bulletin of the Swiss Association of Actuaries* (1970), pp. 111–133. | `src/credibility/buhlmann_straub.rs` — Non-parametric Bühlmann-Straub credibility for non-uniform exposures: Expected Process Variance ($s^2$), Variance of Hypothetical Means ($a$), $K = s^2 / a$, and group weights $Z_i = P_i / (P_i + K)$. | CAS Exam 8 non-uniform exposure credibility test cases. |

---

## 3. Commercial Experience & Retrospective Rating

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **National Council on Compensation Insurance (NCCI)** | NCCI, *Experience Rating Plan Manual for Workers Compensation and Employers Liability Insurance*, Rule 2. | `src/experience/ncci_erp.rs` — Exact NCCI ERP formula $\text{Mod} = (A_p + W A_e + (1 - W) E_e + B) / (E + B)$; split point truncation; ERA 70% medical-only reduction; off-balance adjustment factor $B = \sum E / \sum (E \cdot M)$. | NCCI standard rating worksheet test account. |
| **Lee (1988)** | Y.-S. Lee, *"The Mathematics of Table M"*, *PCAS*, Vol. LXXV (1988), pp. 126–180. | `src/experience/table_m.rs` — Insurance charge $\phi(r) = E[\max(0, Y - r)]$ and savings $\psi(r) = \phi(r) + r - 1$; Lee's differential convexity conditions. | Discrete distribution Table M charge table and convexity validation. |
| **Brosius (1989) / CAS Exam 8** | E. Brosius, *"Table M Construction"*, CAS Study Note (1989); CAS Exam 8 Retrospective Rating syllabus. | `src/experience/retrospective.rs` — Actuarially balanced Retrospective Rating Plan solver: Basic Premium Factor (BPF) iteration under Maximum ($G$) and Minimum ($H$) constraints: $\text{BPF} = e - (\text{LCF} - 1) E + [\phi(r_G) - \psi(r_H)] \cdot \text{LCF} \cdot E$. | Retrospective rating balance convergence test. |

---

## 4. Severity Layering, Deductibles, and Trending

| Source Document | Citation | Implemented Modules & Equations | Verification Benchmark |
| --- | --- | --- | --- |
| **Miccolis (1977)** | R. S. Miccolis, *"An Analysis of Increased Limits Factors"*, *PCAS*, Vol. LXIV (1977), pp. 27–73. | `src/layers/ilf.rs` — Limited Expected Value $E[X \wedge L]$ for Lomax (Pareto Type II), Single-Parameter Pareto, and Exponential distributions; Increased Limits Factors $\text{ILF}(L) = E[X \wedge L] / E[X \wedge B]$; Loss Elimination Ratios $\text{LER}(d) = E[X \wedge d] / E[X \wedge L]$; commercial excess layers. | Closed-form LEV and excess layer pricing parity. |
| **Kallop (1975) / CAS Exam 5** | R. Kallop, *"A Current Look at Workers Compensation Ratemaking"*, *PCAS*, Vol. LXII (1975); Werner & Modlin (2016). | `src/trending/parallelogram.rs`, `src/trending/trend.rs` — Parallelogram on-level factor calculation for historical rate changes; exponential trend model fitting via log-linear regression. | Geometric rate-level change on-leveling benchmark. |
