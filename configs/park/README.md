# Park 式请求模型的城市权重（park_v3）

`gs_demand_mass_45.csv` 给出 45 个候选地面站城市的数据中心数量 `dc_count` 与 IANA 时区 `tz`，供 `requests.distribution_version: park_v3` 使用。

请求模型按 Park et al. (2025, IEEE TNSM) 式 (15)–(17)：

$$\Gamma_{AB}(t)\propto\frac{s(m_A)+s(m_B)}{2}N_AN_B,\qquad s(m)=1+0.5\cos\left(\frac{\pi}{12}(m-14)\right)$$

其中 $m$ 为城市当地时钟小时（由 `tz` 与数据集起点 2023-01-01T00:00Z 换算），$N$ 为数据中心数量。Park 的常数 $0.02\theta$ 改为标定：一天内总泊松到达率的均值等于 `arrival_rate`，因此 `arrival_rate × amount_mean` 仍是平均提供负载。

数据来源：datacentermap.com 各城市市场页标题中的设施数（“X Data Centers - N Facilities”），2026-10-02/03 查阅；Brasília、São Paulo、Mumbai、New Delhi、Sydney、Adelaide 取自所在国家页的市场表。七个城市在该站无独立市场页，用国家总数近似（城市占主导，是上界）：Addis Ababa（埃塞俄比亚 6）、Dakar（塞内加尔 7）、Cairo（埃及 14）、Nairobi（肯尼亚 19）、Casablanca（摩洛哥 15）、Almaty（哈萨克斯坦 17）、Tehran（伊朗 19）。San Francisco 为市区口径（10），硅谷是该站的独立市场。

每行的 `source` 列记录了具体页面。
