"""
Modelo LSTM en PyTorch para series temporales financieras reales.

Idea clave: en lugar de predecir el precio directamente (lo que suele producir un
modelo que solo "copia" el precio de ayer), el modelo predice el *log-retorno*
del día siguiente a partir de una ventana de días anteriores. El precio se
reconstruye como  precio_mañana = precio_hoy * exp(retorno_predicho).

Se compara siempre contra un baseline ingenuo (mañana = hoy) para medir si el
modelo aporta algo de verdad.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

FEATURES = ["ret", "range", "vol"]  # log-retorno, rango intradía, cambio de volumen


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No se encontraron datos para '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance >= 0.2 devuelve columnas multinivel
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    df["ret"] = np.log(df["Close"]).diff()
    df["range"] = np.log(df["High"] / df["Low"])
    vol = df["Volume"].replace(0, np.nan)
    df["vol"] = np.log(vol).diff().fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 200:
        raise ValueError("Muy pocos datos históricos (se necesitan al menos ~200 días)")
    return df


def make_windows(x: np.ndarray, y: np.ndarray, window: int):
    xs, ys = [], []
    for i in range(window, len(x)):
        xs.append(x[i - window:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (batch, seq, hidden)
        return self.head(out[:, -1]).squeeze(-1)  # último paso -> retorno escalado


# --------------------------------------------------------------------------- #
# Entrenamiento + evaluación + pronóstico
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    ticker: str = "AAPL"
    period: str = "5y"
    window: int = 30
    epochs: int = 40
    hidden: int = 64
    layers: int = 2
    lr: float = 1e-3
    horizon: int = 10
    test_frac: float = 0.15
    mc_samples: int = 50
    seed: int = 42


def run(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()

    df = load_prices(cfg.ticker, cfg.period)
    feats = df[FEATURES].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    dates = df.index.strftime("%Y-%m-%d").tolist()

    # Split temporal (nunca aleatorio en series de tiempo)
    n = len(df)
    n_test = max(int(n * cfg.test_frac), 30)
    split = n - n_test

    # Normalización con estadísticas SOLO del train (evita fuga de información)
    mu = feats[:split].mean(axis=0)
    sd = feats[:split].std(axis=0) + 1e-8
    x_all = (feats - mu) / sd
    y_all = x_all[:, 0]  # objetivo: log-retorno escalado del día t

    X, Y = make_windows(x_all, y_all, cfg.window)
    idx = np.arange(cfg.window, n)              # índice del día que se predice
    train_mask = idx < split
    # Validación: último 10% del train
    tr_idx = np.where(train_mask)[0]
    n_val = max(int(len(tr_idx) * 0.1), 20)
    val_sel, tr_sel = tr_idx[-n_val:], tr_idx[:-n_val]
    te_sel = np.where(~train_mask)[0]

    to_t = lambda a: torch.tensor(a, device=device)
    Xtr, Ytr = to_t(X[tr_sel]), to_t(Y[tr_sel])
    Xva, Yva = to_t(X[val_sel]), to_t(Y[val_sel])
    Xte = to_t(X[te_sel])

    model = LSTMForecaster(len(FEATURES), cfg.hidden, cfg.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.HuberLoss(delta=1.0)  # robusta a días extremos

    best_val, best_state, history = math.inf, None, []
    bs = 64
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        train_loss = tot / len(Xtr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()
        sched.step(val_loss)
        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # ---- Evaluación en test (predicción a 1 día, con datos reales hasta ayer) ----
    with torch.no_grad():
        pred_scaled = model(Xte).cpu().numpy().astype(np.float64)
    pred_ret = pred_scaled * sd[0] + mu[0]
    days = idx[te_sel]                       # días predichos
    prev_close = close[days - 1]
    pred_close = prev_close * np.exp(pred_ret)
    true_close = close[days]
    true_ret = feats[days, 0]

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    metrics = {
        "rmse_lstm": rmse(pred_close, true_close),
        "rmse_naive": rmse(prev_close, true_close),
        "mae_lstm": float(np.mean(np.abs(pred_close - true_close))),
        "mae_naive": float(np.mean(np.abs(prev_close - true_close))),
        "mape_lstm": float(np.mean(np.abs(pred_close - true_close) / true_close) * 100),
        "dir_acc": float(np.mean(np.sign(pred_ret) == np.sign(true_ret)) * 100),
        "dir_acc_always_up": float(np.mean(true_ret > 0) * 100),
        "n_test": int(len(days)),
        "n_train": int(len(tr_sel)),
        "device": str(device),
    }

    # ---- Pronóstico futuro recursivo con MC Dropout (incertidumbre) ----
    last_window = x_all[-cfg.window:].copy()
    avg_other = x_all[-cfg.window:, 1:].mean(axis=0)  # features no-retorno: se asume su media reciente
    model.train()  # activa dropout para muestreo Monte Carlo
    paths = np.zeros((cfg.mc_samples, cfg.horizon))
    with torch.no_grad():
        for s in range(cfg.mc_samples):
            w = last_window.copy()
            price = close[-1]
            for h in range(cfg.horizon):
                r_scaled = model(to_t(w[None].astype(np.float32))).item()
                r = r_scaled * sd[0] + mu[0]
                price = price * math.exp(r)
                paths[s, h] = price
                w = np.vstack([w[1:], np.concatenate([[r_scaled], avg_other])])
    model.eval()

    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=cfg.horizon)
    forecast = {
        "dates": future_dates.strftime("%Y-%m-%d").tolist(),
        "mean": paths.mean(axis=0).tolist(),
        "p10": np.percentile(paths, 10, axis=0).tolist(),
        "p90": np.percentile(paths, 90, axis=0).tolist(),
    }

    return {
        "ticker": cfg.ticker.upper(),
        "last_close": float(close[-1]),
        "last_date": dates[-1],
        "history": {"dates": dates, "close": close.tolist()},
        "test": {
            "dates": [dates[d] for d in days],
            "pred": pred_close.tolist(),
            "true": true_close.tolist(),
        },
        "split_date": dates[split],
        "loss": history,
        "metrics": metrics,
        "forecast": forecast,
    }


if __name__ == "__main__":
    import json, sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    r = run(Config(ticker=t, epochs=15))
    print(json.dumps(r["metrics"], indent=2))
    print("forecast:", [round(v, 2) for v in r["forecast"]["mean"]])
