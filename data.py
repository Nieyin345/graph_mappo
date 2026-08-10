## Class entity definitions

class SimConfig:
    """
    Simulation Configuration
    Modify these values to easily control the number of nodes for testing.
    """
    NUM_GS   = 83   # Number of Ground Stations to use (max 83)
    NUM_HAPS = 30   # Number of HAPs to use (max 30)
    NUM_SATS = 30   # Number of Satellites to use (max 30)

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
    """
    STATIONS = [
        # ================= ASIA (亚洲 - 含东亚、中亚、中东) =================
        {"name": "Beijing",   "lat": 39.9042, "lon": 116.4074, "continent": "Asia"}, # 北京
        {"name": "Shanghai",  "lat": 31.2304, "lon": 121.4737, "continent": "Asia"}, # 上海
        {"name": "Changsha",  "lat": 28.2282, "lon": 112.9388, "continent": "Asia"}, # 长沙
        {"name": "Guangzhou", "lat": 23.1291, "lon": 113.2644, "continent": "Asia"}, # 广州
        {"name": "Chengdu",   "lat": 30.5728, "lon": 104.0668, "continent": "Asia"}, # 成都
        {"name": "Urumqi",    "lat": 43.8256, "lon": 87.6168,  "continent": "Asia"}, # 乌鲁木齐
        {"name": "Lhasa",     "lat": 29.6500, "lon": 91.1000,  "continent": "Asia"}, # 拉萨
        {"name": "HongKong",  "lat": 22.3193, "lon": 114.1694, "continent": "Asia"}, # 香港
        {"name": "Tokyo",     "lat": 35.6762, "lon": 139.6503, "continent": "Asia"}, # 东京 (日本)
        {"name": "Kyoto",     "lat": 35.0116, "lon": 135.7681, "continent": "Asia"}, # 京都 (日本)
        {"name": "Osaka",     "lat": 34.6937, "lon": 135.5023, "continent": "Asia"}, # 大阪 (日本)
        {"name": "Seoul",     "lat": 37.5665, "lon": 126.9780, "continent": "Asia"}, # 首尔 (韩国)
        {"name": "Busan",     "lat": 35.1796, "lon": 129.0756, "continent": "Asia"}, # 釜山 (韩国)
        {"name": "Singapore", "lat": 1.3521,  "lon": 103.8198, "continent": "Asia"}, # 新加坡
        {"name": "Bangkok",   "lat": 13.7563, "lon": 100.5018, "continent": "Asia"}, # 曼谷
        {"name": "Jakarta",   "lat": -6.2088, "lon": 106.8456, "continent": "Asia"}, # 雅加达
        {"name": "Mumbai",    "lat": 19.0760, "lon": 72.8777,  "continent": "Asia"}, # 孟买
        {"name": "NewDelhi",  "lat": 28.6139, "lon": 77.2090,  "continent": "Asia"}, # 新德里
        {"name": "Almaty",    "lat": 43.2220, "lon": 76.8512,  "continent": "Asia"}, # 阿拉木图 (中亚-哈萨克斯坦)
        {"name": "Tashkent",  "lat": 41.2995, "lon": 69.2401,  "continent": "Asia"}, # 塔什干 (中亚-乌兹别克斯坦)
        {"name": "Dubai",     "lat": 25.2048, "lon": 55.2708,  "continent": "Asia"}, # 迪拜 (中东)
        {"name": "Riyadh",    "lat": 24.7136, "lon": 46.6753,  "continent": "Asia"}, # 利雅得 (中东-沙特)
        {"name": "Tehran",    "lat": 35.6892, "lon": 51.3890,  "continent": "Asia"}, # 德黑兰 (中东-伊朗)
        {"name": "Jerusalem", "lat": 31.7683, "lon": 35.2137,  "continent": "Asia"}, # 耶路撒冷 (中东-以色列)
        {"name": "Doha",      "lat": 25.2854, "lon": 51.5310,  "continent": "Asia"}, # 多哈 (中东-卡塔尔)

        # ================= EUROPE (欧洲) =================
        {"name": "London",    "lat": 51.5074, "lon": -0.1278,  "continent": "Europe"}, # 伦敦
        {"name": "Paris",     "lat": 48.8566, "lon": 2.3522,   "continent": "Europe"}, # 巴黎
        {"name": "Berlin",    "lat": 52.5200, "lon": 13.4050,  "continent": "Europe"}, # 柏林
        {"name": "Munich",    "lat": 48.1351, "lon": 11.5820,  "continent": "Europe"}, # 慕尼黑
        {"name": "Moscow",    "lat": 55.7558, "lon": 37.6173,  "continent": "Europe"}, # 莫斯科
        {"name": "Rome",      "lat": 41.9028, "lon": 12.4964,  "continent": "Europe"}, # 罗马
        {"name": "Madrid",    "lat": 40.4168, "lon": -3.7038,  "continent": "Europe"}, # 马德里
        {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041,   "continent": "Europe"}, # 阿姆斯特丹
        {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686,  "continent": "Europe"}, # 斯德哥尔摩
        {"name": "Geneva",    "lat": 46.2044, "lon": 6.1432,   "continent": "Europe"}, # 日内瓦
        {"name": "Zurich",    "lat": 47.3769, "lon": 8.5417,   "continent": "Europe"}, # 苏黎世
        {"name": "Vienna",    "lat": 48.2082, "lon": 16.3738,  "continent": "Europe"}, # 维也纳
        {"name": "Istanbul",  "lat": 41.0082, "lon": 28.9784,  "continent": "Europe"}, # 伊斯坦布尔

        # ================= NORTH AMERICA (北美洲) =================
        {"name": "NewYork",   "lat": 40.7128, "lon": -74.0060, "continent": "North_America"}, # 纽约
        {"name": "WashingtonDC","lat": 38.9072,"lon": -77.0369,"continent": "North_America"}, # 华盛顿特区
        {"name": "Boston",    "lat": 42.3601, "lon": -71.0589, "continent": "North_America"}, # 波士顿
        {"name": "Chicago",   "lat": 41.8781, "lon": -87.6298, "continent": "North_America"}, # 芝加哥
        {"name": "Houston",   "lat": 29.7604, "lon": -95.3698, "continent": "North_America"}, # 休斯顿
        {"name": "Denver",    "lat": 39.7392, "lon": -104.9903,"continent": "North_America"}, # 丹佛 (高原)
        {"name": "Seattle",   "lat": 47.6062, "lon": -122.3321,"continent": "North_America"}, # 西雅图
        {"name": "SanFrancisco","lat": 37.7749,"lon": -122.4194,"continent":"North_America"}, # 旧金山
        {"name": "LosAngeles","lat": 34.0522, "lon": -118.2437,"continent": "North_America"}, # 洛杉矶
        {"name": "Miami",     "lat": 25.7617, "lon": -80.1918, "continent": "North_America"}, # 迈阿密
        {"name": "Toronto",   "lat": 43.6510, "lon": -79.3470, "continent": "North_America"}, # 多伦多
        {"name": "Montreal",  "lat": 45.5017, "lon": -73.5673, "continent": "North_America"}, # 蒙特利尔
        {"name": "Vancouver", "lat": 49.2827, "lon": -123.1207,"continent": "North_America"}, # 温哥华
        {"name": "MexicoCity","lat": 19.4326, "lon": -99.1332, "continent": "North_America"}, # 墨西哥城

        # ================= SOUTH AMERICA (南美洲) =================
        {"name": "SaoPaulo",  "lat": -23.5505,"lon": -46.6333, "continent": "South_America"}, # 圣保罗
        {"name": "RioDeJaneiro","lat":-22.9068,"lon":-43.1729, "continent": "South_America"}, # 里约热内卢
        {"name": "BuenosAires","lat": -34.6037,"lon": -58.3816,"continent": "South_America"}, # 布宜诺斯艾利斯
        {"name": "Santiago",  "lat": -33.4489,"lon": -70.6693, "continent": "South_America"}, # 圣地亚哥
        {"name": "Bogota",    "lat": 4.7110,  "lon": -74.0721, "continent": "South_America"}, # 波哥大 (哥伦比亚高海拔区)
        {"name": "Lima",      "lat": -12.0464,"lon": -77.0428, "continent": "South_America"}, # 利马 (秘鲁)
        {"name": "Brasilia",  "lat": -15.7975,"lon": -47.8919, "continent": "South_America"}, # 巴西利亚 (巴西腹地)
        {"name": "Caracas",   "lat": 10.4806, "lon": -66.9036, "continent": "South_America"}, # 加拉加斯 (委内瑞拉)
        {"name": "Quito",     "lat": -0.1807, "lon": -78.4678, "continent": "South_America"}, # 基多 (厄瓜多尔，极度靠近赤道)

        # ================= AFRICA (非洲) =================
        {"name": "Cairo",     "lat": 30.0444, "lon": 31.2357,  "continent": "Africa"}, # 开罗 (埃及)
        {"name": "Lagos",     "lat": 6.5244,  "lon": 3.3792,   "continent": "Africa"}, # 拉各斯 (尼日利亚)
        {"name": "Johannesburg","lat":-26.2041,"lon": 28.0473, "continent": "Africa"}, # 约翰内斯堡 (南非)
        {"name": "CapeTown",  "lat": -33.9249,"lon": 18.4241,  "continent": "Africa"}, # 开普敦 (南非)
        {"name": "Pretoria",  "lat": -25.7479,"lon": 28.2293,  "continent": "Africa"}, # 比勒陀利亚 (南非行政首都)
        {"name": "Nairobi",   "lat": -1.2921, "lon": 36.8219,  "continent": "Africa"}, # 内罗毕 (肯尼亚)
        {"name": "Casablanca","lat": 33.5731, "lon": -7.5898,  "continent": "Africa"}, # 卡萨布兰卡 (摩洛哥)
        {"name": "AddisAbaba","lat": 9.0222,  "lon": 38.7468,  "continent": "Africa"}, # 亚的斯亚贝巴 (埃塞俄比亚)
        {"name": "Dakar",     "lat": 14.7167, "lon": -17.4677, "continent": "Africa"}, # 达喀尔 (塞内加尔，西非海岸)
        {"name": "Algiers",   "lat": 36.7538, "lon": 3.0588,   "continent": "Africa"}, # 阿尔及尔 (阿尔及利亚，北非地中海)
        {"name": "Kinshasa",  "lat": -4.4419, "lon": 15.2663,  "continent": "Africa"}, # 金沙萨 (刚果民主共和国)

        # ================= OCEANIA (大洋洲) =================
        {"name": "Sydney",    "lat": -33.8688,"lon": 151.2093, "continent": "Oceania"}, # 悉尼 (澳大利亚)
        {"name": "Melbourne", "lat": -37.8136,"lon": 144.9631, "continent": "Oceania"}, # 墨尔本 (澳大利亚)
        {"name": "Brisbane",  "lat": -27.4698,"lon": 153.0251, "continent": "Oceania"}, # 布里斯班 (澳大利亚)
        {"name": "Perth",     "lat": -31.9505,"lon": 115.8605, "continent": "Oceania"}, # 珀斯 (澳大利亚西部)
        {"name": "Adelaide",  "lat": -34.9285,"lon": 138.6007, "continent": "Oceania"}, # 阿德莱德 (澳大利亚)
        {"name": "Auckland",  "lat": -36.8485,"lon": 174.7633, "continent": "Oceania"}, # 奥克兰 (新西兰)
        {"name": "Wellington","lat": -41.2865,"lon": 174.7762, "continent": "Oceania"}, # 惠灵顿 (新西兰)
        {"name": "Christchurch","lat":-43.5320,"lon": 172.6362,"continent": "Oceania"}, # 克赖斯特彻奇 (新西兰南岛)
        {"name": "Suva",      "lat": -18.1248,"lon": 178.4501, "continent": "Oceania"}, # 苏瓦 (斐济，太平洋深处)

        # ================= ANTARCTICA (南极洲) =================
        {"name": "McMurdo",   "lat": -77.8463,"lon": 166.6682, "continent": "Antarctica"}, # 麦克默多科考站
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
        # ================= STRATEGIC HAP DEPLOYMENTS (Near GS clusters) =================
        {"name": "HAP_Beijing_Jingjinji", "lat": 39.5, "lon": 116.0, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Shanghai_Yangtze",  "lat": 31.0, "lon": 121.0, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Guangdong_BayArea", "lat": 23.0, "lon": 113.5, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Chengdu_Basin",     "lat": 30.5, "lon": 104.0, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Japan_Kanto",       "lat": 35.5, "lon": 139.0, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Korea_Peninsula",   "lat": 37.0, "lon": 127.5, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Southeast_Asia",    "lat": 1.3,  "lon": 103.8, "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_India_West",        "lat": 19.0, "lon": 73.0,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_MiddleEast_Gulf",   "lat": 25.0, "lon": 55.0,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_Central_Asia",      "lat": 42.0, "lon": 73.0,  "alt_km": 20.0, "region": "Asia"},
        {"name": "HAP_UK_London",         "lat": 51.5, "lon": 0.0,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_France_Paris",      "lat": 48.5, "lon": 2.5,   "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Germany_Central",   "lat": 50.0, "lon": 10.0,  "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Italy_Rome",        "lat": 42.0, "lon": 12.5,  "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_Nordic_Baltic",     "lat": 59.0, "lon": 18.0,  "alt_km": 20.0, "region": "Europe"},
        {"name": "HAP_US_EastCoast",      "lat": 40.0, "lon": -74.5, "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_US_WestCoast",      "lat": 36.0, "lon": -120.0,"alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_US_Midwest",        "lat": 41.5, "lon": -88.0, "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_US_South",          "lat": 29.5, "lon": -95.0, "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_Canada_East",       "lat": 44.0, "lon": -77.0, "alt_km": 20.0, "region": "North_America"},
        {"name": "HAP_Brazil_South",      "lat": -23.0,"lon": -45.0, "alt_km": 20.0, "region": "South_America"},
        {"name": "HAP_Andes_Region",      "lat": -10.0,"lon": -76.0, "alt_km": 20.0, "region": "South_America"},
        {"name": "HAP_South_Africa",      "lat": -28.0,"lon": 26.0,  "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_North_Africa",      "lat": 31.0, "lon": 30.0,  "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_West_Africa",       "lat": 8.0,  "lon": 5.0,   "alt_km": 20.0, "region": "Africa"},
        {"name": "HAP_Australia_East",    "lat": -31.0,"lon": 150.0, "alt_km": 20.0, "region": "Oceania"},
        {"name": "HAP_Australia_South",   "lat": -36.0,"lon": 142.0, "alt_km": 20.0, "region": "Oceania"},
        {"name": "HAP_NewZealand",        "lat": -40.0,"lon": 174.0, "alt_km": 20.0, "region": "Oceania"},
        {"name": "HAP_Polar_Arctic",      "lat": 75.0, "lon": 0.0,   "alt_km": 20.0, "region": "Arctic"},
        {"name": "HAP_Polar_Antarctic",   "lat": -75.0,"lon": 160.0, "alt_km": 20.0, "region": "Antarctic"}
    ]
    
    @classmethod
    def get_haps(cls):
        return cls.HAPS[:SimConfig.NUM_HAPS]

class GlobalSatellites(): ## Global Satellites Database
    """
    A centralized database of satellite initial positions, orbit types, and altitudes.
    """
    SATS = [
        # ================= LEO (Low Earth Orbit) SATELLITES (~500-1200 km) =================
        # Plane 1 (Equatorial)
        {"name": "Sat_LEO_Eq_1", "init_lat": 0.0, "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 500.0, "inclination": 0.0},
        {"name": "Sat_LEO_Eq_2", "init_lat": 0.0, "init_lon": 72.0,  "orbit_type": "LEO", "alt_km": 500.0, "inclination": 0.0},
        {"name": "Sat_LEO_Eq_3", "init_lat": 0.0, "init_lon": 144.0, "orbit_type": "LEO", "alt_km": 500.0, "inclination": 0.0},
        {"name": "Sat_LEO_Eq_4", "init_lat": 0.0, "init_lon": -144.0,"orbit_type": "LEO", "alt_km": 500.0, "inclination": 0.0},
        {"name": "Sat_LEO_Eq_5", "init_lat": 0.0, "init_lon": -72.0, "orbit_type": "LEO", "alt_km": 500.0, "inclination": 0.0},
        
        # Plane 2 (Polar)
        {"name": "Sat_LEO_Pol_1", "init_lat": 90.0, "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 600.0, "inclination": 90.0},
        {"name": "Sat_LEO_Pol_2", "init_lat": 45.0, "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 600.0, "inclination": 90.0},
        {"name": "Sat_LEO_Pol_3", "init_lat": 0.0,  "init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 600.0, "inclination": 90.0},
        {"name": "Sat_LEO_Pol_4", "init_lat": -45.0,"init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 600.0, "inclination": 90.0},
        {"name": "Sat_LEO_Pol_5", "init_lat": -90.0,"init_lon": 0.0,   "orbit_type": "LEO", "alt_km": 600.0, "inclination": 90.0},

        # Plane 3 (Mid-latitude inclined 53 deg)
        {"name": "Sat_LEO_Mid_1", "init_lat": 53.0, "init_lon": 30.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_2", "init_lat": 26.5, "init_lon": 60.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_3", "init_lat": 0.0,  "init_lon": 90.0,  "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_4", "init_lat": -26.5,"init_lon": 120.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_5", "init_lat": -53.0,"init_lon": 150.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_6", "init_lat": -26.5,"init_lon": -150.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_7", "init_lat": 0.0,  "init_lon": -120.0,"orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_8", "init_lat": 26.5, "init_lon": -90.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},
        {"name": "Sat_LEO_Mid_9", "init_lat": 53.0, "init_lon": -60.0, "orbit_type": "LEO", "alt_km": 550.0, "inclination": 53.0},

        # ================= MEO (Medium Earth Orbit) SATELLITES (~10000-20000 km) =================
        {"name": "Sat_MEO_1", "init_lat": 45.0,  "init_lon": 45.0,  "orbit_type": "MEO", "alt_km": 10000.0, "inclination": 55.0},
        {"name": "Sat_MEO_2", "init_lat": 0.0,   "init_lon": 135.0, "orbit_type": "MEO", "alt_km": 10000.0, "inclination": 55.0},
        {"name": "Sat_MEO_3", "init_lat": -45.0, "init_lon": -135.0,"orbit_type": "MEO", "alt_km": 10000.0, "inclination": 55.0},
        {"name": "Sat_MEO_4", "init_lat": 0.0,   "init_lon": -45.0, "orbit_type": "MEO", "alt_km": 10000.0, "inclination": 55.0},
        {"name": "Sat_MEO_5", "init_lat": 30.0,  "init_lon": 90.0,  "orbit_type": "MEO", "alt_km": 15000.0, "inclination": 55.0},
        {"name": "Sat_MEO_6", "init_lat": -30.0, "init_lon": -90.0, "orbit_type": "MEO", "alt_km": 15000.0, "inclination": 55.0},

        # ================= GEO (Geostationary Equatorial Orbit) SATELLITES (~35786 km) =================
        {"name": "Sat_GEO_Asia",     "init_lat": 0.0, "init_lon": 110.0, "orbit_type": "GEO", "alt_km": 35786.0, "inclination": 0.0},
        {"name": "Sat_GEO_Americas", "init_lat": 0.0, "init_lon": -90.0, "orbit_type": "GEO", "alt_km": 35786.0, "inclination": 0.0},
        {"name": "Sat_GEO_Europe",   "init_lat": 0.0, "init_lon": 15.0,  "orbit_type": "GEO", "alt_km": 35786.0, "inclination": 0.0},
        {"name": "Sat_GEO_Pacific",  "init_lat": 0.0, "init_lon": -150.0,"orbit_type": "GEO", "alt_km": 35786.0, "inclination": 0.0},
        {"name": "Sat_GEO_Indian",   "init_lat": 0.0, "init_lon": 65.0,  "orbit_type": "GEO", "alt_km": 35786.0, "inclination": 0.0}
    ]
    
    @classmethod
    def get_satellites(cls):
        return cls.SATS[:SimConfig.NUM_SATS]

