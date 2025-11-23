from __future__ import annotations

from pathlib import Path
from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib import colors as mcolors

from log_psplines.coarse_grain import CoarseGrainConfig
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.mcmc import run_mcmc
from log_psplines.plotting.base import extract_plotting_data
from log_psplines.plotting.psd_matrix import plot_psd_matrix

st.set_page_config(
    page_title="Log-P-Spline Finance Demo",
    layout="wide",
    page_icon="📈",
)

HERE = Path(__file__).resolve().parent
BASE_RESULTS_DIR = HERE.parent / "results" / "finance"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICKER_SECTORS: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "WMT": "Industrial/Retail",
    "CAT": "Industrial/Retail",
    "SPY": "Benchmark",
}

SECTOR_COLORS: dict[str, str] = {
    "Technology": "#3B73B9",
    "Financials": "#9C4F96",
    "Energy": "#EF8A17",
    "Industrial/Retail": "#4DA167",
    "Benchmark": "#7F7F7F",
    "Other": "#999999",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _sector_color(ticker: str) -> str:
    return SECTOR_COLORS.get(
        TICKER_SECTORS.get(ticker, "Other"), SECTOR_COLORS["Other"]
    )


def _with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgba = mcolors.to_rgba(color)
    return (rgba[0], rgba[1], rgba[2], alpha)


def _slugify_run(tickers: Iterable[str], start: str, end: str, label: str | None):
    tick_str = "-".join(t.upper() for t in tickers)
    start_slug = start.replace("-", "")
    end_slug = end.replace("-", "")
    prefix = f"{start_slug}_{end_slug}_{tick_str}"
    return f"{label}_{prefix}" if label else prefix


def _setup_presentation_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f7",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "legend.frameon": False,
            "font.size": 14,
        }
    )


def _build_timeseries(log_returns: pd.DataFrame) -> MultivariateTimeseries:
    y = log_returns.values.astype(np.float64)
    t = np.arange(len(log_returns), dtype=np.float64)
    return MultivariateTimeseries(y=y, t=t)


def _select_time_blocks(n_time: int, min_block_len: int = 128) -> int:
    if n_time <= min_block_len:
        return 1

    max_blocks = max(1, n_time // min_block_len)
    n_blocks = 1
    while (n_blocks * 2) <= max_blocks:
        n_blocks *= 2

    while n_blocks > 1 and n_time % n_blocks != 0:
        n_blocks //= 2

    return max(1, n_blocks)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=False)
    if raw.empty:
        raise ValueError("No data returned for selected tickers/date range.")

    if "Adj Close" in raw.columns.get_level_values(0):
        prices = raw["Adj Close"]
    else:
        prices = raw["Close"]

    return prices


@st.cache_data(show_spinner=False)
def make_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    logr = np.log(prices / prices.shift(1)).dropna()
    return logr - logr.mean()


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def _plot_prices(prices: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    unique_sectors: list[str] = []
    for idx, ticker in enumerate(prices.columns):
        color = _sector_color(ticker)
        linestyle = "-" if idx % 2 == 0 else "--"
        ax.plot(
            prices.index,
            prices[ticker],
            label=ticker,
            alpha=0.95,
            lw=1.8,
            color=color,
            ls=linestyle,
        )
        sector = TICKER_SECTORS.get(ticker, "Other")
        if sector not in unique_sectors:
            unique_sectors.append(sector)

    ax.set_title("Daily Closing Prices", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_yscale("log")
    ax.set_xlim(prices.index.min(), prices.index.max())
    ax.legend(loc="upper left", fontsize=12, ncols=2)
    fig.tight_layout()
    return fig


def _plot_returns(log_returns: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    unique_sectors: list[str] = []
    for idx, ticker in enumerate(log_returns.columns):
        color = _sector_color(ticker)
        linestyle = "-" if idx % 2 == 0 else "--"
        ax.plot(
            log_returns.index,
            log_returns[ticker],
            label=ticker,
            alpha=0.95,
            lw=1.6,
            color=color,
            ls=linestyle,
        )
        sector = TICKER_SECTORS.get(ticker, "Other")
        if sector not in unique_sectors:
            unique_sectors.append(sector)

    ax.set_title("Daily Log-Returns (demeaned)", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return")
    ax.legend(loc="upper left", fontsize=12, ncols=2)
    fig.tight_layout()
    return fig



def _plot_coherence_matrix(
    freqs: np.ndarray,
    coherence_quantiles: np.ndarray,
    percentiles: np.ndarray,
    tickers: list[str],
    save_path: Path,
) -> Path:
    q05 = _get_percentile_slice(coherence_quantiles, percentiles, 5.0)
    q50 = _get_percentile_slice(coherence_quantiles, percentiles, 50.0)
    q95 = _get_percentile_slice(coherence_quantiles, percentiles, 95.0)

    freq_nz = freqs[1:]
    periods = 1.0 / freq_nz

    n = len(tickers)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(3.2 * n, 3.0 * n),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    axes = np.asarray(axes)

    xticks = np.array([5, 30, 180, 365 * 2], dtype=float)
    xtick_labels = np.array(["5d", "1mo", "6mo", "2y"])
    valid = (xticks >= periods.min()) & (xticks <= periods.max())
    xticks = xticks[valid]
    xtick_labels = xtick_labels[valid]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i < j:
                ax.set_axis_off()
                continue
            if i == j:
                ax.set_axis_off()
                ax.text(
                    0.05,
                    0.15,
                    tickers[i],
                    ha="left",
                    va="top",
                    fontsize=18,
                    weight="bold",
                    color=_sector_color(tickers[i]),
                    transform=ax.transAxes,
                )
                continue

            row_color = _sector_color(tickers[i])
            col_color = _sector_color(tickers[j])

            ax.set_facecolor(_with_alpha(row_color, 0.14))
            ax.fill_between(
                periods,
                q05[1:, i, j],
                q95[1:, i, j],
                color=_with_alpha(col_color, 0.3),
            )
            ax.plot(periods, q50[1:, i, j], color=col_color, lw=1.6)
            ax.axhline(0.5, color="#2f2f2f", linestyle=":", linewidth=0.9)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.grid(ls=":", lw=0.7)
            ax.set_xlim(periods.min(), periods.max())

            ax.text(
                0.05,
                0.9,
                tickers[i],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                color=row_color,
                clip_on=False,
            )
            ax.text(
                0.05,
                0.9,
                f"\n× {tickers[j]}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                color=col_color,
                clip_on=False,
            )

            if i != n - 1:
                ax.set_xticklabels([])
            else:
                if xticks.size:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

            if j != 0:
                ax.set_yticklabels([])

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _get_percentile_slice(
    arr: np.ndarray, percentiles: np.ndarray, target: float
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - target)))
    return arr[idx]


def _extract_coherence_quantiles(idata):
    extracted = extract_plotting_data(idata)
    quantiles = extracted.get("posterior_psd_matrix_quantiles")
    if quantiles is None:
        return None, None

    coherence = quantiles.get("coherence")
    if coherence is None:
        return None, None

    percentiles = np.asarray(quantiles["percentile"], dtype=float)
    freqs = extracted.get("frequencies")
    if freqs is None:
        freqs = np.asarray(idata.posterior_psd.coords["freq"], dtype=float)
    else:
        freqs = np.asarray(freqs, dtype=float)
    return freqs, (percentiles, np.asarray(coherence, dtype=float))


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def _extract_median_psd(idata) -> tuple[np.ndarray, np.ndarray]:
    group = None
    if (
        hasattr(idata, "posterior_psd")
        and "psd_matrix_real" in idata.posterior_psd
    ):
        group = idata.posterior_psd
    elif (
        hasattr(idata, "vi_posterior_psd")
        and "psd_matrix_real" in idata.vi_posterior_psd
    ):
        group = idata.vi_posterior_psd
    else:
        raise RuntimeError("InferenceData missing PSD matrix summaries.")

    real_ds = group["psd_matrix_real"]
    imag_ds = group["psd_matrix_imag"]
    percentiles = np.asarray(real_ds.coords["percentile"], dtype=float)
    idx = int(np.argmin(np.abs(percentiles - 50.0)))

    freq = np.asarray(real_ds.coords["freq"], dtype=float)
    psd_real = np.asarray(real_ds.isel(percentile=idx))
    psd_imag = np.asarray(imag_ds.isel(percentile=idx))
    return freq, psd_real + 1j * psd_imag


def estimate_spectral_matrix(
    log_returns: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
    label: str | None,
    *,
    cache_results: bool,
    only_vi: bool,
    vi_steps: int,
    vi_lr: float,
    n_samples: int,
    n_warmup: int,
) -> tuple[np.ndarray, np.ndarray, az.InferenceData, EmpiricalPSD, Path]:
    slug = _slugify_run(tickers, start, end, label)
    results_dir = BASE_RESULTS_DIR / slug
    results_dir.mkdir(parents=True, exist_ok=True)
    idata_path = results_dir / "finance_pspline_inference.nc"

    timeseries = _build_timeseries(log_returns)
    n_time = timeseries.y.shape[0]
    n_blocks = _select_time_blocks(n_time)

    dt = timeseries.t[1] - timeseries.t[0]
    fs = 1.0 / dt
    fmin = fs / (n_time * 1.0)
    fmax = 0.5 * fs

    coarse_cfg = CoarseGrainConfig(
        enabled=False,
        f_transition=5e-2,
        n_log_bins=160,
        f_min=fmin,
        f_max=fmax,
    )

    idata = None
    if cache_results and idata_path.exists():
        idata = az.from_netcdf(idata_path)

    if idata is None:
        idata = run_mcmc(
            data=timeseries,
            sampler="multivar_blocked_nuts",
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_knots=10,
            degree=3,
            diffMatrixOrder=2,
            n_time_blocks=n_blocks,
            coarse_grain_config=coarse_cfg,
            only_vi=only_vi,
            vi_steps=vi_steps,
            vi_lr=vi_lr,
            vi_progress_bar=True,
            rng_key=0,
            verbose=True,
            outdir=str(results_dir),
            fmin=fmin,
            fmax=fmax,
        )
        if cache_results:
            idata.to_netcdf(idata_path)

    freqs, psd_matrix = _extract_median_psd(idata)
    empirical_psd = timeseries.get_empirical_psd()
    return freqs, psd_matrix, idata, empirical_psd, results_dir


# -----------------------------------------------------------------------------
# Streamlit layout
# -----------------------------------------------------------------------------


def _render_sidebar():
    st.sidebar.header("Configuration")

    default_tickers = ["AAPL", "MSFT", "JPM", "BAC", "SPY"]
    tickers = st.sidebar.multiselect(
        "Tickers",
        options=list(TICKER_SECTORS.keys()),
        default=default_tickers,
        help="Choose at least two tickers for coherence plots.",
    )
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2010-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2020-01-01"))
    label = st.sidebar.text_input(
        "Run label (optional)",
        help="Used to namespace cached inference results on disk.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Inference")
    mode = st.sidebar.radio(
        "Mode",
        ["Variational inference (fast)", "Full NUTS"],
        index=0,
    )
    only_vi = mode.startswith("Variational")

    vi_steps = st.sidebar.slider(
        "VI steps",
        min_value=5000,
        max_value=25000,
        value=10000,
        step=1000,
        help="Number of gradient steps when VI-only is selected.",
        disabled=not only_vi,
    )
    vi_lr = st.sidebar.number_input(
        "VI learning rate",
        min_value=1e-5,
        max_value=1e-2,
        value=5e-4,
        step=1e-5,
        format="%g",
        help="Adam step size for VI optimisation.",
        disabled=not only_vi,
    )

    n_samples = st.sidebar.slider(
        "Posterior samples",
        min_value=250,
        max_value=1500,
        value=750,
        step=50,
    )
    n_warmup = st.sidebar.slider(
        "Warmup draws",
        min_value=250,
        max_value=1500,
        value=750,
        step=50,
    )

    cache_results = st.sidebar.checkbox(
        "Use cached inference (if available)", value=True
    )

    return {
        "tickers": tickers,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "label": label.strip() or None,
        "only_vi": only_vi,
        "vi_steps": vi_steps,
        "vi_lr": float(vi_lr),
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "cache_results": cache_results,
    }


def _render_intro() -> None:
    st.title("📈 Log-P-Spline Stock Spectra")
    st.markdown(
        """
        Explore multivariate power spectral densities (PSDs) and coherences for
        daily stock log-returns using the log-P-spline model. Configure tickers,
        date ranges, and variational inference settings from the sidebar, then
        launch the analysis.
        """
    )


def main() -> None:
    _setup_presentation_style()
    config = _render_sidebar()
    _render_intro()

    if not config["tickers"]:
        st.warning("Please select at least one ticker to begin.")
        return

    if st.button("Run analysis", type="primary"):
        with st.spinner("Downloading prices..."):
            prices = download_prices(
                config["tickers"], config["start"], config["end"]
            )

        st.success("Data downloaded")
        st.pyplot(_plot_prices(prices))

        log_returns = make_log_returns(prices)
        st.pyplot(_plot_returns(log_returns))

        with st.spinner("Running log-P-spline inference..."):
            freqs, S, idata, empirical_psd, results_dir = estimate_spectral_matrix(
                log_returns,
                config["tickers"],
                config["start"],
                config["end"],
                config["label"],
                cache_results=config["cache_results"],
                only_vi=config["only_vi"],
                vi_steps=config["vi_steps"],
                vi_lr=config["vi_lr"],
                n_samples=config["n_samples"],
                n_warmup=config["n_warmup"],
            )

        st.success(f"Inference complete. Results cached in {results_dir}.")

        # st.pyplot(_plot_psds_period(freqs, S, config["tickers"]))

        with st.spinner("Rendering coherence plots..."):
            # psd_path = Path(results_dir) / "finance_psd_matrix.png"
            # plot_psd_matrix(
            #     idata=idata,
            #     freq=freqs,
            #     empirical_psd=empirical_psd,
            #     outdir=str(results_dir),
            #     filename="finance_psd_matrix.png",
            #     diag_yscale="log",
            #     offdiag_yscale="linear",
            #     xscale="log",
            #     show_coherence=True,
            #     overlay_vi=True,
            # )
            # st.image(psd_path, caption="PSD matrix with coherence")

            coh_freqs, coh_payload = _extract_coherence_quantiles(idata)
            if coh_payload is not None and coh_freqs is not None:
                percentiles, coherence_quantiles = coh_payload
                coh_path = _plot_coherence_matrix(
                    coh_freqs,
                    coherence_quantiles,
                    percentiles,
                    config["tickers"],
                    Path(results_dir) / "finance_coherence_matrix.png",
                )
                st.image(coh_path, caption="Pairwise coherence bands")
            else:
                st.info("Coherence quantiles unavailable for the current run.")

        st.caption("Spectral matrix shape: %s" % (S.shape,))


if __name__ == "__main__":
    main()