-- Observed conversion lift by decile of a model's predicted uplift (decile 1 = highest).
-- A model that ranks users well shows observed lift falling from decile 1 to decile 10.
-- Input table: scored(treatment, converted, score)
WITH ranked AS (
    SELECT
        treatment,
        converted,
        score,
        NTILE(10) OVER (ORDER BY score DESC) AS decile
    FROM scored
)
SELECT
    decile,
    COUNT(*)                                        AS n_users,
    AVG(score)                                      AS mean_predicted_uplift,
    AVG(converted) FILTER (WHERE treatment = 1)
        - AVG(converted) FILTER (WHERE treatment = 0) AS observed_lift
FROM ranked
GROUP BY decile
ORDER BY decile;
