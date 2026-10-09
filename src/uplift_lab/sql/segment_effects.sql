-- Conversion lift by pre-registered segment, with a normal-approximation standard error.
-- Segments are defined only from pre-treatment attributes, so each comparison is
-- still a randomised experiment within the segment.
-- Input table: experiment(treatment, converted, tenure_days, sessions_7d, channel, country_tier)
WITH labelled AS (
    SELECT
        treatment,
        converted,
        CASE
            WHEN tenure_days < 30 THEN 'new (<30d)'
            WHEN tenure_days < 180 THEN 'established (30-179d)'
            ELSE 'veteran (180d+)'
        END AS tenure_segment,
        CASE WHEN sessions_7d <= 2 THEN 'low activity' ELSE 'active' END AS activity_segment,
        channel,
        'tier ' || CAST(country_tier AS VARCHAR) AS country_segment
    FROM experiment
),
long AS (
    SELECT 'tenure' AS dimension, tenure_segment AS segment, treatment, converted FROM labelled
    UNION ALL
    SELECT 'activity', activity_segment, treatment, converted FROM labelled
    UNION ALL
    SELECT 'channel', channel, treatment, converted FROM labelled
    UNION ALL
    SELECT 'country', country_segment, treatment, converted FROM labelled
),
by_arm AS (
    SELECT
        dimension,
        segment,
        COUNT(*) FILTER (WHERE treatment = 1)          AS n_treatment,
        COUNT(*) FILTER (WHERE treatment = 0)          AS n_control,
        AVG(converted) FILTER (WHERE treatment = 1)    AS rate_treatment,
        AVG(converted) FILTER (WHERE treatment = 0)    AS rate_control
    FROM long
    GROUP BY dimension, segment
)
SELECT
    dimension,
    segment,
    n_treatment,
    n_control,
    rate_control,
    rate_treatment,
    rate_treatment - rate_control AS lift,
    SQRT(
        rate_treatment * (1 - rate_treatment) / n_treatment
        + rate_control * (1 - rate_control) / n_control
    ) AS lift_std_error
FROM by_arm
ORDER BY dimension, lift DESC;
