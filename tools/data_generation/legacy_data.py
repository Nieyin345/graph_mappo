## Class entity definitions

class SimConfig:
    """
    Simulation Configuration
    Modify these values to easily control the number of nodes for testing.
    """
    NUM_GS   = 30   # Number of Ground Stations to use
    NUM_HAPS = 30   # Number of HAPs to use (one per GS)
    NUM_SATS = 30   # Number of Satellites to use

class demand(): ## Key Rate Demand
    def __init__(self, K_REQ, n1, n2):
        self.K_REQ = K_REQ  # Required key rates at each time slot
        self.n1    = n1     # Source node
        self.n2    = n2     # Destination node

class gs(): ## Ground Station
    def __init__(self, lg, la, A_MAX, tag=None):
        self.lg    = lg     # GS longitude (In degrees)
        self.la    = la     # GS latitude  (In degrees)
        self.A_MAX = A_MAX  # Maximum size of QKP
        self.tag   = tag    # HAP's name tag (Optional)
        
class hap(): ## High Altitude Platform
    def __init__(self, lg, la, H, A_MAX, tag=None):
        self.lg    = lg     # HAP longitude list (In degrees - for each time step)
        self.la    = la     # HAP latitude list  (In degrees - for each time step)
        self.H     = H      # HAP altitude
        self.A_MAX = A_MAX  # Maximum size of QKP
        self.tag   = tag    # HAP's name tag (Optional)
        
class sat(): ## LEO Satellite
    def __init__(self, lg, la, H, A_MAX, tag=None):
        self.lg    = lg     # Satellite longitude list (In degrees - for each time step)
        self.la    = la     # Satellite latitude list  (In degrees - for each time step)
        self.H     = H      # Satellite altitude list (In km, ~600)
        self.A_MAX = A_MAX  # Maximum size of QKP
        self.tag   = tag    # Satellite's name tag (Optional)
        
class link(): ## Physical Link (SAT-HAP, HAP-GS, SAT-GS, SAT-SAT)
    def __init__(self, n1, n2, V, W, K_MAX, link_type="hap_gs"):
        self.n1        = n1        # link's source
        self.n2        = n2        # link's destination
        self.V         = V         # Visibility in km (For each time step)
        self.W         = W         # Weather condition (For each time step) (fog, rain, snow)
        self.K_MAX     = K_MAX     # Max link capacities at each time slot
        self.link_type = link_type # "sat_hap", "hap_gs", "sat_gs", or "sat_sat"

class path(): ## Multi-Hop Path (e.g., GS-HAP-GS, GS-SAT-HAP-GS)
    def __init__(self, links):
        self.links = links  # List of links forming the path
        self.hops  = len(links)
        
class system(): ## Other System-Wide Parameters
    def __init__(self, T, THETA, G):
        self.T     = T     # Set of time slots
        self.THETA = THETA # Duration of a time slot
        self.G     = G     # Connectivity matrix

class GlobalGroundStations(): ## Global Ground Station Coordinates Database
    """
    A centralized database of real-world coordinates for global quantum communication ground stations.
    The first ``SimConfig.NUM_GS`` entries are the active stations, selected for roughly uniform
    global coverage (spread across continents, hemispheres and longitude bands).
    """
    STATIONS = [
        # ================= ACTIVE: GLOBALLY UNIFORM 30 GS =================
        # --- North America (5) ---
        {"name": "SanFrancisco","lat": 37.7749, "lon": -122.4194, "continent":"North_America"}, # 旧金山
        {"name": "Vancouver", "lat": 49.2827, "lon": -123.1207, "continent": "North_America"}, # 温哥华
        {"name": "Chicago",   "lat": 41.8781, "lon": -87.6298, "continent": "North_America"}, # 芝加哥
        {"name": "NewYork",   "lat": 40.7128, "lon": -74.0060, "continent": "North_America"}, # 纽约
        {"name": "MexicoCity","lat": 19.4326, "lon": -99.1332, "continent": "North_America"}, # 墨西哥城

        # --- South America (3) ---
        {"name": "Bogota",    "lat": 4.7110,  "lon": -74.0721, "continent": "South_America"}, # 波哥大
        {"name": "SaoPaulo",  "lat": -23.5505,"lon": -46.6333, "continent": "South_America"}, # 圣保罗
        {"name": "Santiago",  "lat": -33.4489,"lon": -70.6693, "continent": "South_America"}, # 圣地亚哥

        # --- Europe (5) ---
        {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686,  "continent": "Europe"}, # 斯德哥尔摩
        {"name": "London",    "lat": 51.5074, "lon": -0.1278,  "continent": "Europe"}, # 伦敦
        {"name": "Moscow",    "lat": 55.7558, "lon": 37.6173,  "continent": "Europe"}, # 莫斯科
        {"name": "Madrid",    "lat": 40.4168, "lon": -3.7038,  "continent": "Europe"}, # 马德里
        {"name": "Rome",      "lat": 41.9028, "lon": 12.4964,  "continent": "Europe"}, # 罗马

        # --- Africa (4) ---
        {"name": "Casablanca","lat": 33.5731, "lon": -7.5898,  "continent": "Africa"}, # 卡萨布兰卡
        {"name": "Lagos",     "lat": 6.5244,  "lon": 3.3792,   "continent": "Africa"}, # 拉各斯
        {"name": "Nairobi",   "lat": -1.2921, "lon": 36.8219,  "continent": "Africa"}, # 内罗毕
        {"name": "Johannesburg","lat":-26.2041,"lon": 28.0473, "continent": "Africa"}, # 约翰内斯堡

        # --- Asia / Middle East (7) ---
        {"name": "Dubai",     "lat": 25.2048, "lon": 55.2708,  "continent": "Asia"}, # 迪拜
        {"name": "Almaty",    "lat": 43.2220, "lon": 76.8512,  "continent": "Asia"}, # 阿拉木图
        {"name": "Mumbai",    "lat": 19.0760, "lon": 72.8777,  "continent": "Asia"}, # 孟买
        {"name": "Singapore", "lat": 1.3521,  "lon": 103.8198, "continent": "Asia"}, # 新加坡
        {"name": "Jakarta",   "lat": -6.2088, "lon": 106.8456, "continent": "Asia"}, # 雅加达
        {"name": "Beijing",   "lat": 39.9042, "lon": 116.4074, "continent": "Asia"}, # 北京
        {"name": "Tokyo",     "lat": 35.6762, "lon": 139.6503, "continent": "Asia"}, # 东京

        # --- Oceania (3) ---
        {"name": "Perth",     "lat": -31.9505,"lon": 115.8605, "continent": "Oceania"}, # 珀斯
        {"name": "Sydney",    "lat": -33.8688,"lon": 151.2093, "continent": "Oceania"}, # 悉尼
        {"name": "Auckland",  "lat": -36.8485,"lon": 174.7633, "continent": "Oceania"}, # 奥克兰

        # --- Europe fill (2 extra for uniform density) ---
        {"name": "Paris",     "lat": 48.8566, "lon": 2.3522,   "continent": "Europe"}, # 巴黎
        {"name": "Berlin",    "lat": 52.5200, "lon": 13.4050,  "continent": "Europe"}, # 柏林

        # --- Antarctica (1) ---
        {"name": "McMurdo",   "lat": -77.8463,"lon": 166.6682, "continent": "Antarctica"}, # 麦克默多科考站

        # ================= RESERVE (unused, kept for reference / valid weather cache) =================
        {"name": "Shanghai",  "lat": 31.2304, "lon": 121.4737, "continent": "Asia"}, # 上海
        {"name": "Changsha",  "lat": 28.2282, "lon": 112.9388, "continent": "Asia"}, # 长沙
        {"name": "Guangzhou", "lat": 23.1291, "lon": 113.2644, "continent": "Asia"}, # 广州
        {"name": "Chengdu",   "lat": 30.5728, "lon": 104.0668, "continent": "Asia"}, # 成都
        {"name": "Urumqi",    "lat": 43.8256, "lon": 87.6168,  "continent": "Asia"}, # 乌鲁木齐
        {"name": "Lhasa",     "lat": 29.6500, "lon": 91.1000,  "continent": "Asia"}, # 拉萨
        {"name": "HongKong",  "lat": 22.3193, "lon": 114.1694, "continent": "Asia"}, # 香港
        {"name": "Kyoto",     "lat": 35.0116, "lon": 135.7681, "continent": "Asia"}, # 京都
        {"name": "Osaka",     "lat": 34.6937, "lon": 135.5023, "continent": "Asia"}, # 大阪
        {"name": "Seoul",     "lat": 37.5665, "lon": 126.9780, "continent": "Asia"}, # 首尔
        {"name": "Busan",     "lat": 35.1796, "lon": 129.0756, "continent": "Asia"}, # 釜山
        {"name": "Bangkok",   "lat": 13.7563, "lon": 100.5018, "continent": "Asia"}, # 曼谷
        {"name": "NewDelhi",  "lat": 28.6139, "lon": 77.2090,  "continent": "Asia"}, # 新德里
        {"name": "Tashkent",  "lat": 41.2995, "lon": 69.2401,  "continent": "Asia"}, # 塔什干
        {"name": "Riyadh",    "lat": 24.7136, "lon": 46.6753,  "continent": "Asia"}, # 利雅得
        {"name": "Tehran",    "lat": 35.6892, "lon": 51.3890,  "continent": "Asia"}, # 德黑兰
        {"name": "Jerusalem", "lat": 31.7683, "lon": 35.2137,  "continent": "Asia"}, # 耶路撒冷
        {"name": "Doha",      "lat": 25.2854, "lon": 51.5310,  "continent": "Asia"}, # 多哈
        {"name": "Paris",     "lat": 48.8566, "lon": 2.3522,   "continent": "Europe"}, # 巴黎
        {"name": "Berlin",    "lat": 52.5200, "lon": 13.4050,  "continent": "Europe"}, # 柏林
        {"name": "Munich",    "lat": 48.1351, "lon": 11.5820,  "continent": "Europe"}, # 慕尼黑
        {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041,   "continent": "Europe"}, # 阿姆斯特丹
        {"name": "Geneva",    "lat": 46.2044, "lon": 6.1432,   "continent": "Europe"}, # 日内瓦
        {"name": "Zurich",    "lat": 47.3769, "lon": 8.5417,   "continent": "Europe"}, # 苏黎世
        {"name": "Vienna",    "lat": 48.2082, "lon": 16.3738,  "continent": "Europe"}, # 维也纳
        {"name": "Istanbul",  "lat": 41.0082, "lon": 28.9784,  "continent": "Europe"}, # 伊斯坦布尔
        {"name": "WashingtonDC","lat": 38.9072,"lon": -77.0369,"continent": "North_America"}, # 华盛顿特区
        {"name": "Boston",    "lat": 42.3601, "lon": -71.0589, "continent": "North_America"}, # 波士顿
        {"name": "Houston",   "lat": 29.7604, "lon": -95.3698, "continent": "North_America"}, # 休斯顿
        {"name": "Denver",    "lat": 39.7392, "lon": -104.9903,"continent": "North_America"}, # 丹佛
        {"name": "Seattle",   "lat": 47.6062, "lon": -122.3321,"continent": "North_America"}, # 西雅图
        {"name": "Toronto",   "lat": 43.6510, "lon": -79.3470, "continent": "North_America"}, # 多伦多
        {"name": "Montreal",  "lat": 45.5017, "lon": -73.5673, "continent": "North_America"}, # 蒙特利尔
        {"name": "LosAngeles","lat": 34.0522, "lon": -118.2437,"continent": "North_America"}, # 洛杉矶
        {"name": "Miami",     "lat": 25.7617, "lon": -80.1918, "continent": "North_America"}, # 迈阿密
        {"name": "RioDeJaneiro","lat":-22.9068,"lon":-43.1729, "continent": "South_America"}, # 里约热内卢
        {"name": "BuenosAires","lat": -34.6037,"lon": -58.3816,"continent": "South_America"}, # 布宜诺斯艾利斯
        {"name": "Lima",      "lat": -12.0464,"lon": -77.0428, "continent": "South_America"}, # 利马
        {"name": "Brasilia",  "lat": -15.7975,"lon": -47.8919, "continent": "South_America"}, # 巴西利亚
        {"name": "Caracas",   "lat": 10.4806, "lon": -66.9036, "continent": "South_America"}, # 加拉加斯
        {"name": "Quito",     "lat": -0.1807, "lon": -78.4678, "continent": "South_America"}, # 基多
        {"name": "Cairo",     "lat": 30.0444, "lon": 31.2357,  "continent": "Africa"}, # 开罗
        {"name": "CapeTown",  "lat": -33.9249,"lon": 18.4241,  "continent": "Africa"}, # 开普敦
        {"name": "Pretoria",  "lat": -25.7479,"lon": 28.2293,  "continent": "Africa"}, # 比勒陀利亚
        {"name": "AddisAbaba","lat": 9.0222,  "lon": 38.7468,  "continent": "Africa"}, # 亚的斯亚贝巴
        {"name": "Dakar",     "lat": 14.7167, "lon": -17.4677, "continent": "Africa"}, # 达喀尔
        {"name": "Algiers",   "lat": 36.7538, "lon": 3.0588,   "continent": "Africa"}, # 阿尔及尔
        {"name": "Kinshasa",  "lat": -4.4419, "lon": 15.2663,  "continent": "Africa"}, # 金沙萨
        {"name": "Melbourne", "lat": -37.8136,"lon": 144.9631, "continent": "Oceania"}, # 墨尔本
        {"name": "Brisbane",  "lat": -27.4698,"lon": 153.0251, "continent": "Oceania"}, # 布里斯班
        {"name": "Adelaide",  "lat": -34.9285,"lon": 138.6007, "continent": "Oceania"}, # 阿德莱德
        {"name": "Wellington","lat": -41.2865,"lon": 174.7762, "continent": "Oceania"}, # 惠灵顿
        {"name": "Christchurch","lat":-43.5320,"lon": 172.6362,"continent": "Oceania"}, # 克赖斯特彻奇
        {"name": "Suva",      "lat": -18.1248,"lon": 178.4501, "continent": "Oceania"}, # 苏瓦
        {"name": "AmundsenScott","lat":-89.9999,"lon": 139.270,"continent": "Antarctica"}  # 阿蒙森-斯科特极点站 (南极点)
    ]
    
    @classmethod
    def get_stations(cls):
        return cls.STATIONS[:SimConfig.NUM_GS]
    
    @classmethod
    def get_by_name(cls, name):
        for s in cls.STATIONS:
            if s["name"].lower() == name.lower():
                return s
        raise ValueError(f"Station {name} not found in global database.")
class GlobalHAPs(): ## Global High Altitude Platforms Database
    """
    A centralized database of coordinates for High Altitude Platforms (HAPs),
    typically floating at ~20km altitude over key regions or oceans to act as relays.
    """
    HAPS = [
        # ================= HAP DEPLOYMENTS (one per active GS, colocated & slight offset) =================
        {"name": "HAP_SanFrancisco", "lat": 38.0, "lon": -122.4, "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_Vancouver",    "lat": 49.3, "lon": -123.1, "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_Chicago",      "lat": 42.0, "lon": -87.6,  "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_NewYork",      "lat": 40.8, "lon": -74.0,  "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_MexicoCity",   "lat": 19.5, "lon": -99.1,  "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_Bogota",       "lat": 4.8,  "lon": -74.0,  "alt_km": 20.0, "region": "South_America"},
        {"name": "HAP_SaoPaulo",     "lat": -23.5,"lon": -46.6,  "alt_km": 20.0, "region": "South_America"},
        {"name": "HAP_Santiago",     "lat": -33.4,"lon": -70.6,  "alt_km": 20.0, "region": "South_America"},
        {"name": "HAP_Stockholm",    "lat": 59.4, "lon": 18.0,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_London",       "lat": 51.6, "lon": -0.1,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Moscow",       "lat": 55.8, "lon": 37.6,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Madrid",       "lat": 40.5, "lon": -3.7,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Rome",         "lat": 42.0, "lon": 12.5,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Casablanca",   "lat": 33.6, "lon": -7.6,   "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_Lagos",        "lat": 6.6,  "lon": 3.4,    "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_Nairobi",      "lat": -1.2, "lon": 36.8,   "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_Johannesburg", "lat": -26.2,"lon": 28.0,   "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_Dubai",        "lat": 25.3, "lon": 55.3,   "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Almaty",       "lat": 43.3, "lon": 76.9,   "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Mumbai",       "lat": 19.1, "lon": 72.8,   "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Singapore",    "lat": 1.4,  "lon": 103.8,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Jakarta",      "lat": -6.2, "lon": 106.8,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Beijing",      "lat": 39.9, "lon": 116.4,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Tokyo",        "lat": 35.7, "lon": 139.6,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Perth",        "lat": -31.9,"lon": 115.8,  "alt_km": 20.0, "region": "Oceania"},
        {"name": "HAP_Sydney",       "lat": -33.8,"lon": 151.2,  "alt_km": 20.0, "region": "Oceania"},
        {"name": "HAP_Auckland",     "lat": -36.8,"lon": 174.7,  "alt_km": 20.0, "region": "Oceania"},
        {"name": "HAP_Paris",        "lat": 48.9, "lon": 2.3,     "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Berlin",       "lat": 52.5, "lon": 13.4,    "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_McMurdo",      "lat": -77.8,"lon": 166.6,  "alt_km": 20.0, "region": "Antarctic"}
    ]
    
    @classmethod
    def get_haps(cls):
        return cls.HAPS[:SimConfig.NUM_HAPS]

class GlobalSatellites(): ## Global Satellites Database
    """
    A centralized database of satellite initial positions, orbit types, and altitudes.
    """
    SATS = [
        # ================= LEO (Low Earth Orbit) SATELLITES — Walker-delta uniform constellation =================
        # All LEO at 550 km with inclination 53 deg (no equatorial / no polar planes).
        # 5 orbital planes (RAAN = init_lon, evenly spaced 72 deg apart); 6 satellites per plane
        # (phase spread via init_lat). This yields near-uniform lat/lon global coverage over time.
        # --- Plane 1 (RAAN 0) ---
        {"name": "Sat_LEO_1a", "init_lat": 42.0, "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_1b", "init_lat": -12.0,"init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_1c", "init_lat": -42.0,"init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_1d", "init_lat": 12.0, "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_1e", "init_lat": 0.0,  "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_1f", "init_lat": 30.0, "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        # --- Plane 2 (RAAN 72) ---
        {"name": "Sat_LEO_2a", "init_lat": 42.0, "init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_2b", "init_lat": -12.0,"init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_2c", "init_lat": -42.0,"init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_2d", "init_lat": 12.0, "init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_2e", "init_lat": 0.0,  "init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_2f", "init_lat": 30.0, "init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        # --- Plane 3 (RAAN 144) ---
        {"name": "Sat_LEO_3a", "init_lat": 42.0, "init_lon": 144.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_3b", "init_lat": -12.0,"init_lon": 144.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_3c", "init_lat": -42.0,"init_lon": 144.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_3d", "init_lat": 12.0, "init_lon": 144.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_3e", "init_lat": 0.0,  "init_lon": 144.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_3f", "init_lat": 30.0, "init_lon": 144.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        # --- Plane 4 (RAAN 216) ---
        {"name": "Sat_LEO_4a", "init_lat": 42.0, "init_lon": -144.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_4b", "init_lat": -12.0,"init_lon": -144.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_4c", "init_lat": -42.0,"init_lon": -144.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_4d", "init_lat": 12.0, "init_lon": -144.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_4e", "init_lat": 0.0,  "init_lon": -144.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_4f", "init_lat": 30.0, "init_lon": -144.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        # --- Plane 5 (RAAN 288) ---
        {"name": "Sat_LEO_5a", "init_lat": 42.0, "init_lon": -72.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_5b", "init_lat": -12.0,"init_lon": -72.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_5c", "init_lat": -42.0,"init_lon": -72.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_5d", "init_lat": 12.0, "init_lon": -72.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_5e", "init_lat": 0.0,  "init_lon": -72.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_5f", "init_lat": 30.0, "init_lon": -72.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0}
    ]
    
    @classmethod
    def get_satellites(cls):
        return cls.SATS[:SimConfig.NUM_SATS]

