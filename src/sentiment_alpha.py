"""
NLP Sentiment Alpha Engine: Financial Headline Sentiment Analysis & Dynamic Tilt.
Extracts sentiment signals from financial news feeds to dynamically rebalance
and scale portfolio asset weights and risk exposure.
"""
from typing import Dict, List, Optional
import numpy as np
import re


class SentimentAlphaEngine:
    """Extracts sentiment polarity from financial news and tilts portfolio weights."""

    # Curated financial lexicon with polarity weights
    FINANCIAL_LEXICON = {
        # Bullish / Positive signals
        "beat": 0.8,
        "surge": 0.85,
        "soar": 0.9,
        "record": 0.7,
        "growth": 0.6,
        "profit": 0.65,
        "rally": 0.8,
        "upgrade": 0.75,
        "outperform": 0.85,
        "dividend": 0.5,
        "expansion": 0.6,
        "bullish": 0.9,
        "breakthrough": 0.8,
        "strong": 0.5,
        "gain": 0.5,
        "optimistic": 0.6,
        # Bearish / Negative signals
        "miss": -0.8,
        "plunge": -0.85,
        "slump": -0.8,
        "decline": -0.6,
        "loss": -0.7,
        "downgrade": -0.75,
        "underperform": -0.8,
        "recession": -0.85,
        "inflation": -0.4,
        "investigation": -0.9,
        "lawsuit": -0.8,
        "bearish": -0.9,
        "default": -0.95,
        "warning": -0.6,
        "layoffs": -0.5,
        "weak": -0.5,
        "debt": -0.4,
    }

    def __init__(self, tilt_intensity: float = 0.25):
        """
        tilt_intensity: Maximum allowed proportional adjustment to portfolio weights
        based on sentiment signals (e.g., 0.25 = max +/- 25% tilt).
        """
        self.tilt_intensity = tilt_intensity

    def score_headline(self, headline: str) -> float:
        """Computes continuous sentiment score [-1.0, 1.0] for a financial headline."""
        tokens = re.findall(r"\b[a-zA-Z]+\b", headline.lower())
        if not tokens:
            return 0.0

        scores = [self.FINANCIAL_LEXICON[t] for t in tokens if t in self.FINANCIAL_LEXICON]
        if not scores:
            return 0.0
        
        # Mean sentiment weighted with non-linear saturation
        raw_score = float(np.mean(scores))
        return float(np.tanh(raw_score * 1.5))

    def evaluate_ticker_sentiment(
        self,
        ticker_headlines: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Aggregate sentiment alpha across multiple headlines per ticker."""
        sentiment_scores = {}
        for ticker, headlines in ticker_headlines.items():
            if not headlines:
                sentiment_scores[ticker] = 0.0
                continue
            scores = [self.score_headline(h) for h in headlines]
            sentiment_scores[ticker] = float(np.mean(scores))
        return sentiment_scores

    def apply_sentiment_tilt(
        self,
        base_weights: Dict[str, float],
        sentiment_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Dynamically adjusts base weights using sentiment alpha tilts:
        w_tilted = w_base * (1 + alpha * sentiment_score)
        Subject to simplex constraint: sum(w_tilted) = 1.0 and w_i >= 0.
        """
        tilted_weights = {}
        for ticker, w in base_weights.items():
            score = sentiment_scores.get(ticker, 0.0)
            # Apply tilt adjustment
            adjustment = 1.0 + (self.tilt_intensity * score)
            tilted_weights[ticker] = max(w * adjustment, 0.0)

        total_weight = sum(tilted_weights.values())
        if total_weight > 0:
            return {k: round(v / total_weight, 4) for k, v in tilted_weights.items()}
        return base_weights
