-- Per-arm experiment summary: sample sizes, conversion, revenue and reward spend.
-- Input table: experiment(treatment, converted, revenue_14d, net_revenue_14d, reward_paid)
SELECT
    treatment,
    COUNT(*)                         AS n_users,
    AVG(converted)                   AS conversion_rate,
    AVG(revenue_14d)                 AS mean_revenue,
    STDDEV_SAMP(revenue_14d)         AS sd_revenue,
    AVG(net_revenue_14d)             AS mean_net_revenue,
    SUM(reward_paid)                 AS total_reward_paid
FROM experiment
GROUP BY treatment
ORDER BY treatment;
