# Verification of the bibliography: addendum

Session of 17 November 2026, second pass. It completes
`references_verification.md` on two items left open there: the Ravu et al. and
Finn and Chacón papers, and the JHTDB forced isotropic turbulence dataset.

Rules followed: every statement was read on a page or file fetched in this
session; the URL is given next to it. Nothing was filled from memory. A point
that could not be read is marked **NOT VERIFIED**. Verbatim fragments are kept
to a few words each (copyright); numbers are given exactly as printed. Values
that I computed from printed numbers are marked *computed*.

## 1. Ravu, Rudman, Metcalfe, Lester, Khakhar (2016)

### Bibliographic record

| Field | Value | Read at |
|---|---|---|
| Authors | Bharath Ravu, Murray Rudman, Guy Metcalfe, Daniel R. Lester, Devang V. Khakhar (this order) | https://api.crossref.org/works/10.1016/j.jcp.2016.07.018 |
| Title | Creating analytically divergence-free velocity fields from grid-based data (the title in the task statement is exact) | same |
| Journal | Journal of Computational Physics | same |
| Volume, pages | 323, 75-94 | same; also the RMIT repository record (start page 75, end page 94, 20 pages) |
| Date | Crossref "issued" and "published-print": 2016-10. Day of the month and online date: none shown | same |
| DOI | 10.1016/j.jcp.2016.07.018 | same; https://doi.org/10.1016/j.jcp.2016.07.018 resolves to the Elsevier item S0021999116303138 |
| Publisher | Crossref: Elsevier BV. RMIT record: Academic Press, copyright "© 2016 Elsevier" | Crossref; https://api.figshare.com/v2/articles/27493134 |
| Preprint | None found. OpenAlex reports `oa_status: closed`, no repository full text; Semantic Scholar reports status CLOSED. No arXiv record surfaced in two searches | https://api.openalex.org/works/doi:10.1016/j.jcp.2016.07.018 ; https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.jcp.2016.07.018 |

Notes on the record. The MaRDI portal page lists the same five authors in the
same order but prints "Publication date 24 June 2019", which is an indexing
date, not the publication date (https://portal.mardi4nfdi.de/wiki/Publication:2424415).
Do not use it. Crossref carries no abstract for this DOI, OpenAlex has none, and
Semantic Scholar states that the abstract was withheld by the publisher.

### Where the abstract was read

The abstract was read in the metadata record of the RMIT University research
repository, through its public API:
https://api.figshare.com/v2/articles/27493134 (field `description`; the record's
own public address is
https://research-repository.rmit.edu.au/articles/journal_contribution/Creating_analytically_divergence-free_velocity_fields_from_grid-based_data/27493134,
whose HTML page answered HTTP 202 with an empty body to a direct fetch and
HTTP 403 to the fetch tool). The record holds no file: it is an abstract-only
deposit.

The publisher page (ScienceDirect) answered HTTP 403. **The full text was not
read.**

### Answers, from the abstract

| Question | Answer | Status |
|---|---|---|
| Representation built | A vector potential, analytic, built with B-splines and C2-continuous: the abstract's words are "a C(2)continuous analytic vector potential". The velocity is its curl, hence a continuous field that is divergence-free analytically | Verified (abstract) |
| Degree of the spline | Not stated in the abstract. C2 continuity is what the abstract states; "cubic" is stated only by a third party (see below) | **NOT VERIFIED** in the paper itself |
| Input data | "discrete 3D velocity data on a regular grid" | Verified (abstract) |
| 2D or 3D | 3D. The abstract says the method is applied to three problems, each in a different coordinate system; the three systems are not named in the abstract | Verified (abstract); names of the coordinate systems NOT VERIFIED |
| Use | Integration of particle trajectories in incompressible flow fields, "robustly and accurately"; the last sentence says the gain in accuracy gives a more accurate identification of Lagrangian coherent structures | Verified (abstract) |
| Mixing | The word does not appear in the abstract. The MaRDI record lists the zbMATH keywords "discrete velocity data", "divergence-free interpolation", "Lagrangian coherent structures", "vector potential", the class "Stokes and related (Oseen, etc.) flows (76D07)", and cited works whose titles are about chaotic advection and chaotic mixing | Mixing as an application: **NOT VERIFIED** (only inferred from titles of cited works) |
| Relation to Finn and Chacón | The abstract says the method is based on the method of Finn and Chacon (2005), its reference [10] | Verified (abstract) |
| Compression, quantisation, entropy coding | No such word in the abstract, in the zbMATH keywords, or in the three third-party descriptions below. The paper presents an interpolation method | Absent from everything read; absence in the full text **NOT VERIFIED** |
| GPU | Not mentioned in the abstract or in any description read | Absent from everything read; full text **NOT VERIFIED** |
| Error against the original grid data | The abstract says the analytic field matches the grid values "almost everywhere", with errors two to four orders of magnitude lower than those of existing methods. This is a comparative, measured statement. No bound is stated in the abstract | Measured error: verified (abstract). A proven or guaranteed error bound: **NOT VERIFIED**, and nothing read suggests one |

### What third parties say about the method (full texts read)

These are descriptions by other authors, not the paper itself.

- Chang, Azevedo, Batty, "Curl-Flow" (arXiv 2104.00867, PDF read at
  https://arxiv.org/pdf/2104.00867), section 2.2.3: Ravu et al. "sought to
  directly construct spline-based vector potentials"; they set one component of
  the potential to zero; their velocity and potential unknowns are colocated at
  nodes; they solve for the coefficients of continuous cubic spline functions;
  and they solve a global linear system, which that paper calls expensive.
- Schroeder, Roy Chowdhury, Shinar, "Local divergence-free polynomial
  interpolation on MAC grids", J. Comput. Phys. (2022), author manuscript read
  at https://par.nsf.gov/servlets/purl/10357284: Ravu et al. fit the derivatives
  of the spline function for the potential to the velocities and solve a large
  global system for the spline coefficients. The same paper cites Ravu et al.
  for more accurate tracking of Lagrangian trajectories and Lagrangian coherent
  structures over long times.
- Yang, del-Castillo-Negrete, Zhang, Beidler, "A divergence-free constrained
  magnetic field interpolation method for scattered data" (arXiv 2207.12349,
  PDF read at https://arxiv.org/pdf/2207.12349): places Ravu et al. in the
  "potential approach" and says it improves the inversion of the curl operator
  from grid data and the reconstruction of the field from the interpolated
  potential.

None of the three mentions compression, quantisation, entropy coding or a GPU
in connection with Ravu et al.

### Consequence for the article

The statement in `references_verification.md` that the abstract was not
verified can be lifted: representation, input and use are now read. What must
still not be asserted without reading the paper: the spline degree, the
absence of any error bound, and the absence of compression or GPU work. The
safe wording is that Ravu et al. build a C2 B-spline vector potential from
gridded 3D velocities for trajectory integration, and that their abstract
reports measured interpolation errors, not a bound, and no compression.

## 2. Finn and Chacón (2005)

### Bibliographic record

| Field | Value | Read at |
|---|---|---|
| Authors | John M. Finn, Luis Chacón (Los Alamos National Laboratory, T-15, Plasma Theory) | https://api.crossref.org/works?query.bibliographic=Volume+preserving+integrators+for+solenoidal+fields+on+a+grid+Finn+Chacon ; https://aipp.silverchair-cdn.com/article-minimal/1015543 (redirect target of the AIP article page) |
| Title | Volume preserving integrators for solenoidal fields on a grid | same |
| Journal | Physics of Plasmas 12 (5), article 054503 (2005). Type: Brief Report | same |
| Dates | Received 10 January 2005, accepted 15 February 2005, published 18 April 2005 | AIP page |
| DOI | **10.1063/1.1889156** | Crossref, AIP page, https://www.osti.gov/biblio/20722118 |

A search-result summary proposed the DOI 10.1063/1.1914766 as "likely": it is
wrong, do not use it.

### Does the abstract describe a spline representation of a vector potential?

**No, not in those words.** The abstract was read at three places (Crossref,
OSTI record 20722118, AIP page), with identical text. Its relevant sentence
says that the method involves an interpolation scheme which keeps the field
solenoidal throughout the domain, on and off the grid, "by means of tricubic
splines". The words "vector potential" do not occur in the abstract. The AIP
page fetched shows the abstract and the reference list only; the body of the
article is behind the paywall. **That the tricubic spline is applied to a
vector potential is NOT VERIFIED in the paper's own text.**

It is stated by four later papers whose text was read:

- Schroeder et al. (2022), https://par.nsf.gov/servlets/purl/10357284: Finn and
  Chacón "reconstruct a tricubic spline representation of the vector potential".
- Yang et al. (2022), https://arxiv.org/pdf/2207.12349: the reference
  interpolates the vector potential with local tricubic splines and takes the
  analytic curl of the spline formulas.
- Beznosov, Bonilla, Wimmer, Tang (arXiv 2501.01523,
  https://arxiv.org/pdf/2501.01523): first a discrete vector potential is built
  on the grid points from the discrete field, then it is fitted with splines to
  give a globally smooth potential, then the field is its analytical curl.
- Ravu et al. (2016), abstract: their B-spline vector potential method is
  "based on the method of Finn and Chacon (2005)".

Safe wording for the article: cite Finn and Chacón for tricubic-spline
interpolation that keeps a gridded solenoidal field solenoidal off the grid
(their abstract), and attribute "spline representation of the vector potential"
to the way later authors describe it, unless the paper itself is read.

## 3. JHTDB forced isotropic turbulence, 1024^3 (`isotropic1024coarse`)

### Documents read

| Short name | Document | URL |
|---|---|---|
| README-2016 | "FORCED ISOTROPIC TURBULENCE DATA SET (EXTENDED)", 4 pages, PDF created 1 December 2016 | https://turbulence.pha.jhu.edu/docs/README-isotropic.pdf |
| README-2026 | "FORCED ISOTROPIC TURBULENCE DATA SET", 3 pages, PDF created 12 February 2026, linked from the current dataset page as "README-isotropic1024" | https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic1024.pdf |
| Old page | Dataset description page, last update 2 December 2019 | https://turbulence.pha.jhu.edu/Forced_isotropic_turbulence.aspx |
| New page | Dataset description page | https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic |
| Li et al. | Li, Perlman, Wan, Yang, Meneveau, Burns, Chen, Szalay, Eyink, arXiv 0804.1703 (DOI 10.1080/14685240802376389), full text | https://arxiv.org/pdf/0804.1703 |
| Announcements | JHTDB announcements | https://turbulence.idies.jhu.edu/announcements ; https://turbulence.pha.jhu.edu/announcements.aspx |

The address https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/forcedIsotropic
given in the task returns HTTP 404; the right path ends in `/isotropic`.

The identifier string `isotropic1024coarse` is now verified: the new site's list
page (https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence) prints
"Isotropic Turbulence on 1,024^3 mesh" with the identifier `isotropic1024coarse`
under it, and the announcement of 9 October 2018 uses the same string.

Both READMEs were also rendered to images and read visually, because the plain
text extraction drops the Greek letters. This settles the point left open in
`references_verification.md`: the domain factor is pi.

### Parameters

| Quantity | Value as printed | Where |
|---|---|---|
| Domain | 2π x 2π x 2π, "(i.e. range of x1, x2 and x3 is [0,2π])" | README-2016 and README-2026 (read on the rendered page). Li et al.: "a [0, 2π]^3-domain" |
| Grid | 1024^3, periodic | both READMEs, Li et al. |
| Method | Pseudo-spectral; viscous term integrated analytically with an integrating factor; other terms second-order Adams-Bashforth; nonlinear term in vorticity form | both READMEs, Li et al. |
| Dealiasing | "de-aliased using phase-shift and a 2√2/3 truncation" (the README cites Patterson and Orszag, 1971) | both READMEs (formula read on the rendered page) |
| kmax | "kmax = 1024√2/3 ≈ 482", called the effective maximum wavenumber, after a "2√2/3 spherical truncation" | Li et al., section 2.1. **Neither README gives kmax.** The spectrum file https://turbulence.pha.jhu.edu/spectrum.txt ends at k = 482 |
| kmax·η | **Not given in any document read.** *Computed*: 482 x 0.00287 = 1.38 (window 0 to 2.048); 482 x 0.00280 = 1.35 (window 0 to 10.056) | NOT VERIFIED as a printed value |
| Viscosity | ν = 0.000185 | both READMEs, Li et al. |
| DNS time step | Δt = 0.0002 | both READMEs, Li et al. |
| Time between stored frames | δt = 0.002, "every 10 DNS time-steps is stored" | both READMEs, both pages, Li et al. |
| Forcing | Total energy kept constant in the modes of wavenumber magnitude at most 2 | both READMEs, both pages |
| Stored frames | README-2016 and both pages: 5028 frames, t from 0 to 10.056. **README-2026: 5,024 frames, t from 0 to 10.048.** The documents disagree by four frames | see "Discrepancies" |
| Stored variables | README-2016 and pages: three velocity components and pressure. README-2026 adds the force | READMEs |
| Duration | About five large-eddy turnover times | both READMEs |

### Statistics, by averaging window

| Quantity | Window t in [0, 10.056] | Window t in [0, 2.048] |
|---|---|---|
| Printed in | README-2016 only | README-2016, README-2026, Li et al. (Table 1) |
| Total kinetic energy | 0.705 | 0.695 |
| Dissipation ε | 0.103 | 0.0928 |
| Rms velocity u' | 0.686 | 0.681 |
| Taylor microscale λ | 0.113 | 0.118 |
| Taylor-scale Reynolds number | **418** | **433** |
| Kolmogorov time scale | 0.0424 | 0.0446 |
| Kolmogorov length scale η | **0.00280** | **0.00287** |
| Integral scale L | 1.364 | 1.376 |
| Large-eddy turnover time | 1.99 | 2.02 |

Both dataset pages give a single figure: the Taylor-scale Reynolds number
"fluctuates around" 433.

*Computed* from these numbers, with a grid spacing of 2π/1024 = 0.0061359:
η is 0.468 grid spacing for the short window and 0.456 for the long one (the
grid spacing is 2.14 η and 2.19 η). No document read prints this ratio.

### How the velocity is stored

| Point | What was read | Status |
|---|---|---|
| Space of storage | The fields are "generated in physical space and ingested into the database" | Li et al., section 2.1 |
| Grid points | Both pages: DNS "using 1,024^3 nodes". Li et al.: "1024^3 grid points in a periodic box". The query documentation offers "none" as spatial option, the value at the datapoint closest to the requested coordinate (https://turbulence.idies.jhu.edu/database/query/methods) | The data are on the collocation points of the pseudo-spectral grid, which the pages call nodes. No page uses the words "cell centre" |
| Coordinates of the points | The announcement of 9 October 2018 names the last planes as "x=2π-dx=6.277049 (i=Nx)" and the first as i=1. *Computed*: 2π minus 2π/1024 = 6.277049, so dx = 2π/1024 and the 1024 planes run from x = 0 to x = 2π - dx | An explicit formula x_i = (i-1)·2π/1024 is **NOT VERIFIED** as printed; it is the only reading consistent with the announcement |
| Precision | **NOT VERIFIED.** No document read states single or double precision for the stored data. The C client interface takes and returns `float` arrays (Li et al., Figure 7; https://turbulence.pha.jhu.edu/help/c-fortran/), which is the type of the interface, not a statement about storage | NOT VERIFIED |
| Layout on disk | Z-order (Morton) index; "atoms" holding a 64^3 cube plus a replicated edge of 4 points per side, 72^3 points each; 27 terabytes for the first 1024 frames | Li et al., section 2 (this describes the 2008 system; whether the current store is identical is NOT VERIFIED) |
| Derivatives | The simulation enforces zero divergence spectrally; the database derivative tools are finite differences or splines, so a non-negligible divergence error is measured through them. README-2026 puts the gradient error at "about 7% rms error" at this resolution | README-2016 note 3; README-2026 page 2 |

### Discrepancies and warnings to carry into the article

1. **Two sets of statistics.** The Reynolds number 433 and η = 0.00287 belong
   to the first 2.048 time units (the original 1024 frames); 418 and 0.00280
   are the averages over the full 10.056 time units, printed only in the 2016
   README. State the window with the number. The 2026 README dropped the
   long-window table.
2. **Frame count.** 5028 frames to t = 10.056 (README-2016, both web pages) or
   5,024 frames to t = 10.048 (README-2026). Not resolved by any page read.
3. **Ingest fault after t = 2.048.** The announcement of 9 October 2018 reports
   an ingest bug affecting `isotropic1024coarse` for t > 2.048: the stored data
   on the last x plane and the last y plane had been set equal to the data on
   the first planes. The announcement of 10 November 2018 says the bugs in
   isotropic1024 have been fixed. Relevant if a cutout predates the fix.
4. **kmax·η is not a printed value.** If the article quotes 1.38 or 1.35 it must
   say that it is computed from kmax ≈ 482 (Li et al.) and η of the README.
5. **Storage precision is not documented** on any page read.

## Pages that could not be read

| Address | Result |
|---|---|
| https://www.sciencedirect.com/ (Ravu et al. article page) | HTTP 403; full text not read |
| https://research-repository.rmit.edu.au/articles/journal_contribution/.../27493134 | HTTP 202 empty body, then HTTP 403; the same record was read through api.figshare.com |
| https://ui.adsabs.harvard.edu/abs/2016JCoPh.323...75R/abstract and .../2005PhPl...12e4503F/abstract | HTTP 405 |
| https://katalog.lib.cas.cz/KNAV/EdsRecord/... | no response |
| https://www.bohrium.com/paper-details/creating-analytically-divergence-free-velocity-fields-from-grid-based-data/... | page protected by a script challenge; no readable text |
| https://zbmath.org/ record of Ravu et al. | HTTP 403 |
| https://pubs.aip.org/ (Finn and Chacón, article body) | abstract and references only; body behind the paywall |
| https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/forcedIsotropic | HTTP 404 (correct path: `/isotropic`) |
| https://turbulence.idies.jhu.edu/cutout/ | HTTP 503 |
