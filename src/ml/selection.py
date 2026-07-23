"""Portfolio selection layer: which (asset, horizon) combos have shown a
real, backtested edge worth actually acting on -- not just statistical
significance -- and what the trained model says to do about them today.

Motivated by the plateau across v7/v9: adding more data sources kept
raising the count of statistically significant findings (48 -> 56), but
the walk-forward backtest win rate stayed flat at 11/39 both times. Most
of the 39 (asset, horizon) combos have never shown a real edge no matter
how many features get thrown at them. Rather than keep treating all 39
equally, this surfaces only the ones with an actual backtest track
record, so a recommendation means something instead of being one of 39
predictions with unknown reliability.
"""
from __future__ import annotations

from src.config.watchlist_loader import load_watchlist
from src.ml.backtest import HORIZONS, choose_model_name
from src.ml.predict import predict_horizon
from src.ml.storage import load_backtest_result


def get_qualified_combos(min_trades: int = 10) -> list[dict]:
    """(asset, horizon) combos where the walk-forward backtest of the
    model actually chosen for trading (the best statistically significant
    model, or a naive fallback that by construction can't beat buy-and-
    hold) beat buy-and-hold, with enough trades behind it that it isn't a
    fluke from a handful of lucky calls.

    Requires a persisted backtest result (src.ml.backtest must have run
    and saved one via storage.save_backtest_result) -- combos with none
    yet are skipped, not treated as disqualified.
    """
    assets = [a for a in load_watchlist() if a.active]
    qualified = []
    for asset in assets:
        for horizon in HORIZONS:
            model_name = choose_model_name(asset.id, horizon)
            result = load_backtest_result(asset.id, horizon, model_name)
            if result is None:
                continue
            if result["beats_buy_hold"] and result["n_trades"] >= min_trades:
                qualified.append(result)
    return qualified


def get_recommendations(min_trades: int = 10) -> list[dict]:
    """Today's live prediction for every qualified combo, with its
    backtest track record attached -- the output actually meant to be
    acted on, instead of the full 39-combo grid where most entries have
    no demonstrated edge behind them.
    """
    recommendations = []
    for combo in get_qualified_combos(min_trades=min_trades):
        pred = predict_horizon(combo["investment_id"], combo["horizon"], combo["model_name"])
        recommendations.append(
            {
                **pred,
                "direction": "alta" if pred["predicted_change_pct"] > 0 else "baixa",
                "backtest_strategy_return": combo["strategy_cumulative_return"],
                "backtest_buy_hold_return": combo["buy_hold_cumulative_return"],
                "backtest_n_trades": combo["n_trades"],
                "backtest_n_blocks": combo["n_blocks"],
            }
        )
    recommendations.sort(key=lambda r: -r["backtest_strategy_return"])
    return recommendations


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)
    for rec in get_recommendations():
        print(
            f"{rec['investment_id']:6s} {rec['horizon']:8s} {rec['direction']:6s} "
            f"previsto={rec['predicted_change_pct']:+.2f}% "
            f"(historico: estrategia={rec['backtest_strategy_return']*100:+.1f}% "
            f"vs buy&hold={rec['backtest_buy_hold_return']*100:+.1f}% "
            f"em {rec['backtest_n_trades']}/{rec['backtest_n_blocks']} operacoes)"
        )
