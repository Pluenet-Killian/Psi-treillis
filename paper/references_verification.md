# Verification of the bibliography

Session of 17 November 2026. Every statement below was read on a page fetched
during that session (publisher or DOI registry record, arXiv, official project
page, institutional repository). Nothing was filled from memory. A field that
could not be read on a fetched page is marked **NOT VERIFIED**.

Sources used most often:

- `api.crossref.org/works/<DOI>`: the Crossref registry record of a DOI
  (title, authors, venue, volume, issue, pages, year). Publisher pages of IEEE,
  ACM, AIP and Elsevier were not fetched; the Crossref record stands for them.
- `arxiv.org/abs/<id>`: arXiv abstract page (versions, dates, journal
  reference, related DOI).
- Project pages and README files on GitHub, for the paper each software asks
  users to cite.

"Abstract not read" in the last column means that the bibliographic fields are
verified but the one-sentence description rests on the title or on a project
page, not on the abstract.

## Table of works

| Key | What the work is | URL verified | Not verified |
|---|---|---|---|
| `liang2023sz3` | Introduces SZ3, a modular framework in which prediction-based error-bounded lossy compressors are composed from interchangeable components, with automatic choice of the best predictor per data block. | https://api.crossref.org/works/10.1109/TBDATA.2022.3201176 ; abstract via the arXiv API record 2111.02925 ; cited by https://github.com/szcompressor/SZ3 as "the software engineering design of SZ3" | Nothing |
| `zhao2021optimizing` | Paper that the SZ3 README names for "SZ3's interpolation-based algorithm" (error-bounded compression by dynamic spline interpolation). | https://api.crossref.org/works/10.1109/ICDE51399.2021.00145 ; https://github.com/szcompressor/SZ3 | Abstract not read. Volume: none shown |
| `liu2024high` | Paper that the SZ3 README names for QoZ v2, "the enhanced interpolation-based algorithm" (auto-tuned multi-component interpolation). | https://api.crossref.org/works/10.1145/3639259 ; https://github.com/szcompressor/SZ3 | Abstract not read. Article number: none shown, pages given as 1-27 |
| `li2023lossy` | Describes SPERR, a lossy compressor for scientific floating-point data that uses wavelet transforms, SPECK coding and a custom outlier coding algorithm to bound the error (wording of the project README). | https://api.crossref.org/works/10.1109/IPDPS54959.2023.00104 ; https://github.com/NCAR/SPERR ("If SPERR benefits your work, please kindly cite this publication") | Abstract not read (description from the README) |
| `lindstrom2014fixed` | The paper that the zfp README asks to cite for scholarly use of zfp (fixed-rate compressed floating-point arrays). | https://api.crossref.org/works/10.1109/TVCG.2014.2346458 ; https://github.com/LLNL/zfp | Abstract not read |
| `diffenderfer2019error` | Error analysis of zfp; the zfp README names it as the description of the algorithm implemented in the current zfp version. | https://api.crossref.org/works/10.1137/18M1168832 ; https://github.com/LLNL/zfp | Abstract not read |
| cuZFP (no entry) | CUDA execution policy of zfp. The zfp documentation states: "The CUDA implementation supports only fixed-rate mode and will fail if other compression modes are specified." | https://zfp.readthedocs.io/en/latest/execution.html ; https://zfp.readthedocs.io/en/latest/introduction.html | **NOT VERIFIED**: any separate publication, author list or repository for cuZFP. The name "cuZFP" does not appear on the zfp pages fetched; https://github.com/LLNL/cuZFP returned HTTP 404. Cite `lindstrom2014fixed` and the zfp software for the CUDA backend |
| `huang2023cuszp` | The first of four papers the cuSZp README asks to cite: a GPU error-bounded lossy compression framework optimised for end-to-end performance (title). | https://api.crossref.org/works/10.1145/3581784.3607048 ; https://github.com/szcompressor/cuSZp | Abstract not read |
| `huang2024cuszp2` | Second paper named by the cuSZp README (cuSZp2). | Crossref record via https://api.crossref.org/works?query.bibliographic=cuSZp2+A+GPU+Lossy+Compressor+with+Extreme+Throughput+and+Optimized+Compression+Ratio ; https://github.com/szcompressor/cuSZp | Abstract not read. Crossref prints the title as "CUSZP2: ..."; the README prints "cuSZp2: ...". The README also lists two 2025 papers (Huang et al., SC'25, "GPU Lossy Compression for HPC Can Be Versatile and Ultra-Fast"; Zhang et al., ICS'25, "Pushing the Limits of GPU Lossy Compression: A Hierarchical Delta Approach"), not entered here: their DOIs are NOT VERIFIED |
| `pluzhnik2026property` | Lossy compression of divergence-free velocity fields that keeps zero divergence (and periodicity on periodic domains) for any compression tolerance, by fitting nodal velocities with property-preserving matrix-valued kernels and compressing the kernel coefficients with standard lossy compressors. | https://arxiv.org/abs/2610.02776 ; https://arxiv.org/html/2610.02776v1 | Nothing. Current version: **v1, Fri 2 Oct 2026 04:04:49 UTC**, the only version listed. Subject math.NA. No journal reference, no comments field |
| `ni2025representing` | Represents flow fields with divergence-free kernels (matrix-valued radial basis functions derived from Wendland's C4 polynomial); the abstract lists data compression among the experiments. | https://arxiv.org/abs/2504.01913 | Whether coefficients are quantised or entropy-coded, and any error bound: not stated on the abstract page; full text not read |
| `deriaz2006divergence` | Presents wavelet bases with zero divergence for incompressible flows, with fast algorithms in two and three dimensions, and a method that splits a flow into an incompressible part and a gradient part. | https://api.crossref.org/works/10.1080/14685240500260547 ; https://arxiv.org/abs/cs/0502092 | Issue number: none shown (article N3 of volume 7). The arXiv record has a different title ("Divergence-free Wavelets for Navier-Stokes") and the same DOI |
| `chang2022curl` | Adds pointwise divergence-free velocity interpolation to grid-based fluid solvers through a fast reconstruction of a vector potential, with a ramping method for non-penetrating boundaries around irregular obstacles. | https://api.crossref.org/works/10.1145/3550454.3555498 ; https://arxiv.org/abs/2104.00867 | Article number **NOT VERIFIED** (Crossref gives pages 1-21 and no article-number field) |
| `finn2005volume` | Integrates solenoidal flows given on a grid by splitting them into two-dimensional area-preserving flows (Crank-Nicolson or modified leapfrog), and uses tricubic splines to keep the field solenoidal on and off the grid. | https://api.crossref.org/works/10.1063/1.1889156 (abstract included) ; OSTI record 20722118 | The abstract read does not contain the words "vector potential"; that the spline is applied to a vector potential is **NOT VERIFIED** from a fetched page (it is stated only in a search-result summary of the Ravu et al. record) |
| `ravu2016creating` | Journal article on building analytically divergence-free velocity fields from grid-based data. | https://api.crossref.org/works/10.1016/j.jcp.2016.07.018 | Abstract **NOT VERIFIED** on a fetched page (the repository page returned HTTP 403). A search-result summary describes a B-spline method that computes a C2-continuous analytic vector potential from discrete 3D velocity data on a regular grid, building on Finn and Chacón; read the paper before citing it for that |
| `treib2012turbulence` | A GPU system that compresses turbulence flow fields with a wavelet scheme including run-length and entropy coding, decodes them on the graphics card inside brick-based ray-casting, and reaches about 5 seconds per time step on a desktop PC. | https://api.crossref.org/works/10.1109/TVCG.2012.274 ; https://cs.cit.tum.de/cg/research/publications/2012/turbulence-visualization-at-the-terascale-on-desktop-pcs | Nothing. Note: the paper has **six** authors (Treib, Bürger, Reichl, Meneveau, Szalay, Westermann); Alex Szalay is missing from the list in the task statement. Crossref prints "K. Burger"; the TUM page prints "Kai Bürger" |
| `treib2015analyzing` | Studies how lossy compression affects the accuracy of particle trajectories in turbulent vector fields; the effect was minor in some experiments, and the authors state that the results are not generally valid. This is the separate particle-tracing paper. | https://api.crossref.org/works/10.5220/0005307202790288 ; https://www.cs.cit.tum.de/cg/research/publications/2015/analyzing-the-effect-of-lossy-compression-on-particle-traces-in-turbulent-vector-fields/ | Nothing (conference: IVAPP 2015, Berlin, 11-14 March 2015) |
| `li2008public` | Describes a public web-services database of a direct numerical simulation of forced isotropic turbulence on a 1024^3 grid with 1024 time samples (about 27 TB), in which users run remote analyses and request subsets, with differentiation and interpolation. | https://api.crossref.org/works/10.1080/14685240802376389 ; arXiv API record 0804.1703 ; https://turbulence.pha.jhu.edu/citing.aspx | Issue number: none shown (article N31 of volume 9; the JHTDB citing page writes "9, No. 31, 2008") |
| `jhtdbisotropic` | The forced isotropic turbulence dataset of the Johns Hopkins Turbulence Databases; parameters in the section below. | https://turbulence.pha.jhu.edu/docs/README-isotropic.pdf ; https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic ; https://turbulence.pha.jhu.edu/citing.aspx | Year of the dataset record; the identifier string `isotropic1024coarse` (see below) |
| `wu2020data` | Proposes to reduce the size of turbulence databases by sub-sampling in space and time and re-simulating local sub-domains. | https://arxiv.org/abs/1910.11994 | Publisher record not fetched; journal data come from the arXiv journal reference "Phys. Rev. Fluids 5, 064607 (2020)". Issue number NOT VERIFIED |
| `glaws2020deep` | Journal article on deep learning for in situ compression of large turbulent flow simulations (title). | https://api.crossref.org/works/10.1103/PhysRevFluids.5.114602 | Abstract **NOT VERIFIED** (not read on a fetched page); the description is the title only |
| `momenifar2022physics` | Applies a physics-informed deep learning model based on vector quantisation, with convolutional layers, to compress simulated three-dimensional turbulent flows, and reports a compression ratio of 85. | https://api.crossref.org/works/10.1109/DCC52660.2022.00039 ; https://arxiv.org/abs/2201.03617 | Nothing. The journal counterpart (arXiv 2103.01074, "Dimension Reduced Turbulent Flow Data From Deep Vector Quantizers", Journal of Turbulence 2022, DOI 10.1080/14685248.2022.2060508) has volume and pages NOT VERIFIED and is not entered |
| `gong2023mgard` | Main MGARD reference: a multigrid framework for error-controlled data compression and refactoring (title); listed under "MGARD Foundations" in the project README. | https://api.crossref.org/works/10.1016/j.softx.2023.101590 ; https://github.com/CODARcode/MGARD | Abstract not read. The README has no explicit "please cite" sentence; it lists publications |
| `ainsworth2019multilevel` | The MGARD theory paper on quantitative control of accuracy in derived quantities (title). | https://api.crossref.org/works/10.1137/18M1208885 | Abstract not read |
| `di2025survey` | Survey of error-bounded lossy compression for scientific data: six classic models, ten common components, 46 compressors assessed, and application-specific designs. | https://arxiv.org/abs/2404.02840 (v3, 9 April 2025) ; Crossref record 10.1145/3733104 | Nothing. The arXiv page does not yet show the journal reference; the journal record (ACM Computing Surveys 57(11), 1-38, 2025) was read in Crossref |
| `zhao2020sdrbench` | Introduces a benchmark for lossy compressors on real scientific datasets: datasets, quality metrics, integrated compressors, and six lessons from the evaluation. | https://arxiv.org/abs/2101.03201 ; https://sdrbench.github.io/ | **Pages NOT VERIFIED**. The project page gives the venue as the International Workshop on Big Data Reduction (IWBDR2020), held with IEEE BigData 2020 |
| `liang2020toward` | Error-bounded lossy compression that preserves critical points in 2D and 3D piecewise linear vector fields, using per-vertex error bounds in a modified compressor. | https://api.crossref.org/works/10.1109/PacificVis48177.2020.6431 ; https://scholarsmine.mst.edu/comsci_facwork/1123 | The name "cpSZ" does not appear on the pages fetched: **NOT VERIFIED** as the name of this compressor |
| `liang2023toward` | Journal version of the previous paper (title without "2D and 3D"). | Crossref record 10.1109/TVCG.2022.3214821 | Abstract not read |
| `duda2013asymmetric` | Proposes asymmetric numeral systems, an entropy coding method aiming at the speed of Huffman coding with the compression rate of arithmetic coding. | https://arxiv.org/abs/1311.2540 | No journal reference exists on the page (preprint only, "24 pages, 12 figures"). The word "rANS" was not checked in the full text |
| `unser1993bspline1` | Part I (theory) of "B-spline signal processing". | Crossref record 10.1109/78.193220 | Abstract not read; author given names appear only as initials |
| `unser1993bspline2` | Part II (efficiency design and applications) of "B-spline signal processing". | Crossref record 10.1109/78.193221 | Abstract not read; author given names appear only as initials |
| `unser1999splines` | Tutorial article "Splines: a perfect fit for signal and image processing". | Crossref record 10.1109/79.799930 | Abstract not read; author given name appears only as an initial. Which of the three Unser references is "the standard one" for the article is an editorial choice |

Count: 31 BibTeX entries. 28 have every bibliographic field of the entry read
on a fetched record; 3 are partial (`zhao2020sdrbench`: pages; `jhtdbisotropic`:
year; `wu2020data`: publisher record not fetched). One requested item, cuZFP,
has no entry because no publication could be verified.

## The JHTDB dataset used

Read in the README of the dataset
(https://turbulence.pha.jhu.edu/docs/README-isotropic.pdf, text extracted from
the PDF) and on the dataset page
(https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic).

| Quantity | Value as printed | Where |
|---|---|---|
| Simulation | Direct numerical simulation of forced isotropic turbulence, pseudo-spectral, periodic grid | README, dataset page |
| Grid | 1024^3 | README, dataset page |
| Domain | printed as "2 x 2 x 2 (i.e. range of x1, x2 and x3 is [0,2])" in the extracted text | README. **The factor is NOT VERIFIED**: a glyph (most probably the letter pi) was lost in the text extraction of the PDF |
| Viscosity | 0.000185 | README |
| Simulation time step | 0.0002 | README |
| Time between stored frames | 0.002, "every 10 DNS time-steps is stored" | README, dataset page |
| Stored frames | 5028, t between 0 and 10.056, three velocity components and pressure | README, dataset page |
| Taylor-scale Reynolds number | 418 averaged over t in [0, 10.056]; 433 averaged over t in [0, 2.048] | README. The dataset page says it "fluctuates around" 433 |
| Kolmogorov length | 0.00280 averaged over [0, 10.056]; 0.00287 averaged over [0, 2.048] | README |
| Kolmogorov time | 0.0424 averaged over [0, 10.056]; 0.0446 averaged over [0, 2.048] | README |
| Dissipation | 0.103 ([0, 10.056]); 0.0928 ([0, 2.048]) | README |
| rms velocity | 0.686 ([0, 10.056]); 0.681 ([0, 2.048]) | README |
| Integral scale, large-eddy turnover time | 1.364 and 1.99 ([0, 10.056]); 1.376 and 2.02 ([0, 2.048]) | README |
| Dataset DOI | 10.7281/T1KK98XB | citing page |

Points to mind when quoting these figures in the article:

- The README gives **two** sets of statistics (two averaging windows). The
  Reynolds number is 418 or 433, and the Kolmogorov length 0.00280 or 0.00287,
  according to the window. State which one is used.
- The identifier `isotropic1024coarse` was **NOT VERIFIED**: the pages fetched
  call the dataset "coarse" (frames stored every 10 DNS steps) and "fine"
  (every step), and do not print the identifier string.
- Requested citations (https://turbulence.pha.jhu.edu/citing.aspx): Li et al.
  2008 (`li2008public`), and E. Perlman, R. Burns, Y. Li and C. Meneveau,
  "Data Exploration of Turbulence Simulations using a Database Cluster",
  Supercomputing SC07, ACM, IEEE, 2007 (pages and DOI NOT VERIFIED, not
  entered in the .bib), plus the dataset DOI above.
- Note 3 of the README: the simulation enforces zero divergence spectrally,
  and the divergence evaluated with finite differences on the stored data is
  "non-negligible".

## Closest prior work

**A. Pluzhnik, R. Sharma, M. Burtscher, V. Shankar, "Property-Preserving Data
Compression for Incompressible Flows", arXiv:2610.02776, version 1 of
2 October 2026** (the only version on 17 November 2026). Everything below was
read in the abstract page and in the HTML full text
(https://arxiv.org/html/2610.02776v1), through two passes of an automatic
reader; section numbers are those it reported and should be rechecked against
the PDF before being quoted.

- **Representation.** The velocity is a linear combination of matrix-valued
  divergence-free kernel functions, obtained by differentiating a scalar
  radial kernel (Matérn and Wendland kernels). What is stored is the vector of
  kernel coefficients. Node coordinates, centre indices and kernel parameters
  are shared and are not counted in the stored bytes. Five fitting schemes:
  global Matérn interpolation, two-level Nyström, Nyström plus Wendland hybrid,
  multilevel Wendland interpolation, and a frame kernel method. The text does
  not use a vector potential or a stream function, and B-splines are not
  mentioned. The alternative it discusses is a Helmholtz-Hodge projection
  after decompression, which "requires an additional solve".
- **Compression of the coefficients.** Done by existing error-bounded lossy
  compressors applied to the coefficient arrays: ZFP, PFPL, SZ3 and MGARD on
  the CPU; CUDA ZFP, PFPL-CUDA, cuSZ and MGARD-X on the GPU. No quantiser or
  entropy coder of their own is described. Each velocity field is compressed
  separately; no temporal prediction.
- **What is proved.** (1) A preservation lemma: every decompressed coefficient
  vector defines a field that satisfies the linear constraints (zero
  divergence, periodicity where applicable), "independently of the compression
  tolerance". (2) A nodal error bound in the **discrete Euclidean (l2) norm
  over the stored nodes**: total nodal error is at most the fitting error plus
  a weighted sum of the coefficient perturbations, so that a suitable choice
  of coefficient tolerances "limits the additional nodal error to τ". The
  bound is absolute; a relative error is obtained by dividing by the Euclidean
  norm of the nodal velocity vector. The fitting error is a separate term that
  the bound does not control. There is **no pointwise (maximum-norm) guarantee
  at every node**.
- **Data.** Three small unstructured node sets: a 2D lid-driven cavity with
  9,566 nodes on a triangular mesh; a 2D periodic Taylor-Green flow with 7,288
  distinct periodic nodes; a 3D species-transport flow in a Kenics mixer with
  11,388 nodes. No regular grid of DNS size (nothing like 1024^3) appears. The
  paper states that the CPU and GPU runs "establish no speed advantage".
- **Particle tracing and arbitrary-point GPU sampling.** Particle tracing,
  pathlines and Lagrangian tracking do not appear. The kernel expansion is a
  continuous function, so off-node evaluation is possible in principle, but
  the experiments evaluate the reconstruction at the original nodes. GPU use
  covers the compressors (device pointers, compressed sizes returned to the
  host) and the fitting and evaluation in JAX; sampling at arbitrary points on
  the GPU is not described.
- **Novelty claim.** The text read does not claim to be the first compression
  method that preserves incompressibility exactly.
- **Code.** Public: https://github.com/VarShankar/property-preserving-compression
  (repository fetched; public; Python; "no datasets or compiled binaries are
  distributed"; licence not seen on the fetched page).

Consequence for the article: the statement "lossy compression whose decoded
field is divergence-free by construction, for any tolerance" **is already made
by this preprint** (six weeks before this session). What the fetched text of
the preprint does not contain: a vector potential, B-splines on a regular
grid, integer coefficients with a dedicated entropy coder, temporal
prediction, a maximum-error bound checked on every sample, data of DNS size,
GPU sampling and particle tracing.

Other works close to the representation, not to the compression:

- **Ravu, Rudman, Metcalfe, Lester, Khakhar 2016** (`ravu2016creating`):
  bibliographic record verified; according to a search-result summary (not a
  fetched page) it builds a C2 B-spline vector potential from 3D velocity data
  on a regular grid and uses it for particle trajectories. If the summary is
  right, this is the **same representation** as the article's, without any
  compression. It must be read and cited.
- **Finn and Chacón 2005** (`finn2005volume`): tricubic splines used to keep a
  gridded field solenoidal on and off the grid, for volume-preserving
  integration.
- **Chang, Partono, Azevedo, Batty 2022** (`chang2022curl`): vector potential
  reconstructed from a discretely divergence-free grid velocity, interpolated
  and differentiated to obtain a pointwise incompressible field.
- **Ni, Xing, Li, Wang, Chen 2025** (`ni2025representing`): divergence-free
  kernels as a compact representation of flow fields, with data compression
  among the experiments; no bound is stated on the abstract page.
- **Deriaz and Perrier 2006** (`deriaz2006divergence`): divergence-free
  wavelet bases in 2D and 3D.

## Searches for prior art

**Result stated plainly.** No work found in these searches stores an
incompressible velocity field (or a magnetic field) as a quantised,
entropy-coded vector potential or stream function, and none combines a
divergence-free decoded field with a pointwise error bound. **One work does
part of what the article claims**: Pluzhnik et al. (arXiv:2610.02776,
2 October 2026) is a lossy compressor whose decoded field is divergence-free
by construction for any tolerance, with a proved (Euclidean-norm) link between
the compressor tolerance and the velocity error. The article cannot claim to
be the first compressor with a divergence-free decoded field; it can only
claim what differs (see the previous section).

**Absence in these searches is not proof of absence.** The searches were web
searches in English, sixteen queries, on one day. They do not cover theses,
patents, conference posters, non-English literature, paywalled full texts, or
codes that store a vector potential without a paper saying so.

### Queries run

1. `lossy compression divergence-free velocity field vector potential quantization entropy coding error bound`
2. `compression of incompressible flow data "stream function" lossy compressor "divergence-free" reconstruction guaranteed`
3. `magnetic field data compression MHD "vector potential" lossy compression preserve "div B" = 0 error-bounded`
4. `fluid simulation velocity field compression computer graphics "divergence-free" compressed representation "vector potential" OR "stream function" quantized`
5. `"divergence-free" wavelet compression turbulence velocity field "compression ratio" thresholding coefficients incompressible DNS data`
6. `"vector potential" compress magnetic field simulation output storage "divergence-free" by construction "lossy" plasma OR tokamak OR stellarator OR solar`
7. `error-bounded lossy compression "divergence" preserving vector field compressor SZ "incompressibility" constraint guarantee decompressed data`
8. `"B-spline" "vector potential" OR "stream function" compact representation turbulent velocity field "divergence-free" data reduction particle tracking GPU`
9. `"Compressing Fluid Subspaces" Jones Sen Kim divergence-free basis compression fluid animation`
10. `neural implicit representation compression flow field "divergence-free" by construction curl of "vector potential" network scientific data compression`
11. `"solenoidal" lossy compression velocity field "Helmholtz" OR "vector potential" compressor decompressed field "exactly divergence-free" pointwise error bound`
12. `lossy compression magnetic field data "divergence-free" decompressed "div B" compression MHD simulation snapshots zfp SZ divergence error after decompression`
13. `"physics-preserving" OR "constraint-preserving" OR "structure-preserving" lossy compression scientific data "divergence-free" incompressible velocity 2025 2026`
14. `compress velocity field by storing "vector potential" instead of velocity "compression" "divergence-free" turbulence database particle tracking spline coefficients quantized integers`
15. `fluid animation "velocity field" compression "divergence-free" codec "stream function" OR "vector potential" SIGGRAPH smoke simulation data compression playback`
16. `"vector potential" data compression OR "compressed storage" magnetic field "divergence-free" "lossy" MHD turbulence dataset OR "particle-in-cell" quantization "div B" preserved exactly`

### Relevant hits whose page was fetched

| Work | URL | What the fetched page says | Queries |
|---|---|---|---|
| Pluzhnik, Sharma, Burtscher, Shankar, 2026 | https://arxiv.org/abs/2610.02776 | Lossy compression of divergence-free velocity fields that keeps zero divergence for any tolerance, through divergence-free matrix-valued kernels whose coefficients are compressed by standard lossy compressors. | 13 |
| Ni, Xing, Li, Wang, Chen, 2025 | https://arxiv.org/abs/2504.01913 | Divergence-free kernels as a representation of flow fields, with experiments "spanning data compression, inpainting, super-resolution, and time-continuous flow inference". | 2, 4, 10 |
| Chang, Partono, Azevedo, Batty, 2022 | https://arxiv.org/abs/2104.00867 | Pointwise divergence-free interpolation for grid-based fluids through a reconstructed vector potential; an interpolation method, not a compressor. | 2, 4, 11, 15 |
| Finn and Chacón, 2005 | https://api.crossref.org/works/10.1063/1.1889156 | Volume-preserving integration of gridded solenoidal flows, with tricubic splines keeping the field solenoidal; no compression. | 8 (through Ravu et al.) |
| Ravu et al., 2016 | https://api.crossref.org/works/10.1016/j.jcp.2016.07.018 | Bibliographic record only; the abstract was not read on a fetched page. | 8 |
| Deriaz and Perrier, 2006 | https://arxiv.org/abs/cs/0502092 | Divergence-free wavelet bases for incompressible flows in 2D and 3D, with fast algorithms. | 5, 14 |
| Beznosov, Bonilla, Tang, Wimmer, 2025 (plasma) | https://arxiv.org/abs/2501.01523 | Hermite interpolation that reconstructs divergence-free magnetic fields from a vector potential for particle trajectories and Poincaré sections; data compression is not mentioned. | 6 |
| del-Castillo-Negrete, Hirshman, Spong, D'Azevedo, 2007 (MHD) | https://www.osti.gov/biblio/20991563-compression-magnetohydrodynamic-simulation-data-using-singular-value-decomposition | Compression of MHD field data by singular value decomposition, judged on electron-orbit Poincaré plots; the page mentions neither divergence-free fields nor a vector potential (Journal of Computational Physics 222(1), 2007, DOI 10.1016/j.jcp.2006.07.022). | 12 |
| Wu, Zaki, Meneveau, 2020 | https://arxiv.org/abs/1910.11994 | Reduction of turbulence databases by space-time sub-sampling and local re-simulation; a different principle. | 14 |
| Treib, Bürger, Wu, Westermann, 2015 | https://www.cs.cit.tum.de/cg/research/publications/2015/analyzing-the-effect-of-lossy-compression-on-particle-traces-in-turbulent-vector-fields/ | Effect of lossy compression on particle traces in turbulent fields. | 8 |
| Momenifar, Diao, Tarokh, Bragg, 2022 | https://arxiv.org/abs/2201.03617 | Vector-quantised autoencoder for 3D turbulence, physics-informed; incompressibility is a constraint of the training, not a property of the decoder by construction (the latter half of this sentence comes from the search-result summary, not from the abstract page). | 4 |
| Liang et al., 2020 and 2023 | https://scholarsmine.mst.edu/comsci_facwork/1123 | Error-bounded compression of vector fields that preserves critical points; divergence is not addressed. | 1, 7, 11 |

### Hits seen in search results only (page not fetched: statements NOT VERIFIED)

| Work | URL returned by the search | What the search result says | Queries |
|---|---|---|---|
| Chang, Azevedo, Batty, "Divergence-Free and Boundary-Respecting Velocity Interpolation Using Stream Functions", 2019 | https://diglib.eg.org/handle/10.1145/3309486-3339890 (HTTP 403 when fetched) | Velocity built as the curl of a stream function for interpolation in grid fluids. | 2, 4 |
| "Deep Fluids: A Generative Network for Parameterized Fluid Simulations" | https://arxiv.org/pdf/1806.02071 | A generative network that outputs a stream function whose curl is the velocity, hence divergence-free by construction; a learned reduced representation of simulations, with no error bound. | 2 |
| Jones, Sen, Kim, "Compressing Fluid Subspaces", 2016 | https://diglib.eg.org/handle/10.2312/sca20161225 | Transform codec for the basis matrices of subspace fluid simulation, exploiting the sparsity of Laplacian eigenfunctions in the frequency domain; compression "up to an order of magnitude". Not a divergence-free guarantee. | 9 |
| In situ autoencoder compression with a divergence term in the loss | https://ar5iv.labs.arxiv.org/html/2210.09262 | Divergence-free behaviour encouraged by a penalty during training, not guaranteed. | 2 |
| "FaCTz: Fast Critical-Point and Topology-Aware GPU Compression for Scientific Vector Fields" | https://arxiv.org/html/2608.10586 | GPU error-bounded compressor preserving critical points of vector fields; divergence not addressed. | 7 |
| "Toward Quantity-of-Interest Preserving Lossy Compression for Scientific Data" | https://www.vldb.org/pvldb/vol16/p697-liang.pdf | Error guarantees on derived quantities of interest; divergence not addressed in the snippet. | 13 |
| Ando, Thuerey, Wojtan, stream function solver for liquids, 2015 | https://pub.ista.ac.at/~wojtan/projects/2015_Ando_ASFSfLS/download/vecpotential.pdf | A solver in vector-potential form; simulation, not compression. | 4, 15 |
| "Divergence-free Interpolation of Vector Fields From Point Values - Exact divB=0 in Numerical Simulations" | https://arxiv.org/pdf/1102.4852 | Divergence-free interpolation of magnetic fields; mentions codes that store and interpolate the vector potential. Simulation and interpolation, not compression. | 3 |
| MHD solvers evolving the vector potential (several) | https://arxiv.org/abs/1812.01701 ; https://arxiv.org/abs/2609.35989 ; https://arxiv.org/abs/1712.08922 | The vector potential as the evolved variable, giving div B = 0 to machine precision; none concerns storage or lossy compression. | 3, 6, 16 |
| "Divergence-free wavelet analysis of turbulent flows", 2002 | https://portal.mardi4nfdi.de/wiki/Item:Q1610515 | Divergence-free wavelets applied to a 2D DNS flow and to PIV slices, including a study of compression properties. The closest wavelet precedent for compressing in a divergence-free basis; authors and venue NOT VERIFIED. | 5 |

### Reading of the searches

- Scientific-data compression literature (SZ, zfp, MGARD, SPERR families, the
  2025 survey): the guarantees found concern pointwise error, critical points,
  topology, spectra or quantities of interest. Apart from Pluzhnik et al., no
  compressor found guarantees a divergence-free decoded field.
- MHD and plasma literature: the vector potential is widely used as the
  evolved or interpolated variable to keep div B = 0 (solvers, field-line and
  particle tracing). No paper found uses it as a lossy storage format with
  quantisation and entropy coding.
- Computer-graphics literature: stream functions and vector potentials are
  used for interpolation (Curl-Flow and its 2019 predecessor), for solvers,
  and as the output of generative networks; the one codec found (Jones, Sen,
  Kim 2016) compresses subspace bases and gives no divergence guarantee.
- Divergence-free wavelets (Deriaz and Perrier; the 2002 analysis paper) are
  the older route to sparse divergence-free representations of turbulence;
  the pages fetched do not describe an error-bounded codec built on them. The
  full texts were not read, so a thresholding-based compression experiment in
  those papers is possible and should be checked before the article states
  that none exists.
