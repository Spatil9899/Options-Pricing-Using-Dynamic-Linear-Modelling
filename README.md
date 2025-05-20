# Options-Pricing-Using-Dynamic-Linear-Modelling
Bayesian Options Pricing — GARCH-based volatility, Kalman state-space filtering, and full posterior uncertainty analysis in Python.
What we built — Think of it as a smart autopilot for option pricing. We feed yesterday’s market noise into a volatility engine (GARCH 1,1), let a classic Black-Scholes formula spit out theoretical prices, and then hand everything to a Kalman filter that keeps nudging the forecasts each day as fresh quotes roll in. The three pieces work together like weather-forecast models sharing data streams.

Who & why — Two IIT finance grads (Vinayak Mettu and Sourabh Patil) turned a spring-2025 independent study with Prof. Bruce Rawlings into a proof-of-concept for “live” Bayesian option pricing. We wanted to see whether academic tricks can survive contact with real markets—spoiler: mostly yes, but with caveats.

Real-world testbed — Six months of data (Nov 1 2024 → Mar 21 2025) covering 37 large-cap stocks, daily T-bill yields from FRED, and full option chains from Polygon.io/Bloomberg. Everything lines up on the same trading calendar so the code runs straight out of the box.

How well it works — Our filter’s prices tracked the market with a jaw-dropping 0.99 correlation and an R² of 0.9896—meaning it explained basically all day-to-day price moves in the sample. Regression slopes came in near 1.0, so the model wasn’t just accurate; it was also pretty unbiased. 

But… reality bites — Option desks don’t blindly trust models; they “stretch” volatility into a smile so they don’t get crushed by tail events. Because we stick to pure stats, our engine under-prices deep in- / out-of-the-money contracts. Takeaway: statistics ≠ trading intuition. 


Sandbox crash-tests — We also generated synthetic price paths (AR + GARCH) and synthetic option chains to make sure the Kalman layer can recover known dynamics before risking live capital. It passed with flying colours. 


Where it’s headed next — Blend in the market’s own implied-vol surface, penalise forecasts that drift too far from traded quotes, and maybe bolt on jumps or stochastic-vol extensions. In other words: make the model as adaptive as the traders it’s trying to emulate. 
