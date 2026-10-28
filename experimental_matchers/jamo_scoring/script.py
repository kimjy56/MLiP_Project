# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import OSA

USE_TRAIN_LOOKUP: bool = True
KR_MODE: str = "AR06"
EN_MODE: str = "EN_V3"

ROMAN_CHAR_MAP = {
    "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5",
    "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
    "ⅰ": "1", "ⅱ": "2", "ⅲ": "3", "ⅳ": "4", "ⅴ": "5",
}
ROMAN_TOKEN_MAP = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
}
EN_NUM_WORD_MAP = {
    "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5",
    "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10",
    "FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4", "FIFTH": "5",
    "1ST": "1", "2ND": "2", "3RD": "3", "4TH": "4", "5TH": "5",
}

CHOSUNG = ["G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
JUNGSUNG = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WO", "WE", "WI", "YU", "EU", "UI", "I"]
JONGSUNG = ["", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "P", "L", "L", "P", "L", "M", "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T"]

CHOSUNG_PHON = ["K", "K", "N", "T", "T", "R", "M", "P", "P", "S", "S", "", "J", "J", "C", "K", "T", "P", "H"]
JUNGSUNG_PHON = ["A", "E", "YA", "YE", "O", "E", "YO", "YE", "O", "WA", "WE", "WE", "YO", "U", "WO", "WE", "WI", "YU", "U", "I", "I"]
JONGSUNG_PHON = ["", "K", "K", "K", "N", "N", "N", "T", "R", "K", "M", "P", "R", "R", "P", "R", "M", "P", "P", "T", "T", "N", "T", "T", "K", "T", "P", "T"]

KR_LETTER_MAP_BASE = {
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E",
    "에프": "F", "쥐": "G", "지": "G", "에이치": "H", "아이": "I", "제이": "J",
    "케이": "K", "엘": "L", "엠": "M", "엔": "N", "오": "O", "피": "P",
    "큐": "Q", "알": "R", "아르": "R", "에스": "S", "티": "T", "유": "U", "브이": "V",
    "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z",
}

KR_LETTER_MAP = {
    **KR_LETTER_MAP_BASE,
    "앤": "N", "앰": "M", "앨": "L", "애스": "S", "애프": "F", "액스": "X",
    "애이": "A", "애이치": "H", "재이": "J", "지트": "Z",
}
KR_LETTER_ITEMS_SORTED = sorted(KR_LETTER_MAP.items(), key=lambda x: -len(x[0]))

KR_EN_NUM_WORD_MAP = {
    "원": "1", "투": "2", "쓰리": "3", "스리": "3", "포": "4", "파이브": "5",
    "식스": "6", "세븐": "7", "에잇": "8", "나인": "9", "텐": "10",
    "퍼스트": "1", "세컨드": "2", "써드": "3",
}

IDEA_B_SYLLABLE_VARS: Dict[str, List[str]] = {
    "현": ["HYUN", "HYEON", "HYON", "XIAN"],
    "정": ["JUNG", "JEONG", "CHUNG", "ZHENG"],
    "신": ["SHIN", "SIN", "XIN"],
    "성": ["SUNG", "SEONG", "SONG", "SHENG", "CHENG"],
    "송": ["SONG", "SOUNG"],
    "영": ["YOUNG", "YEONG", "YONG", "RYONG", "YING"],
    "경": ["KYUNG", "GYEONG", "GYOUNG", "KYONG", "JING"],
    "광": ["KWANG", "GWANG", "GUANG"],
    "진": ["JIN", "CHIN", "ZHEN"],
    "선": ["SUN", "SEON", "SHUN", "XUAN"],
    "동": ["DONG", "TONG"],
    "창": ["CHANG", "CANG"],
    "대": ["DAE", "DAI", "DA"],
    "태": ["TAE", "TAI"],
    "우": ["WOO", "U", "WU"],
    "유": ["YOO", "YU", "YOU", "LIU"],
    "주": ["JOO", "JU", "ZHOU", "ZHU"],
    "수": ["SOO", "SU", "SHU", "XIU"],
    "구": ["KOO", "GU", "OU"],
    "부": ["BOO", "BU", "FU"],
    "두": ["DOO", "DU", "TU"],
    "무": ["MOO", "MU", "WU"],
    "금": ["KUM", "GEUM", "GUM", "KIM", "JIN"],
    "은": ["EUN", "UN"],
    "박": ["PARK", "BAK"],
    "최": ["CHOI", "CHOE", "CUI"],
    "이": ["LEE", "YI", "I", "LI", "E"],
    "김": ["KIM", "GIM"],
    "강": ["KANG", "GANG", "JIANG"],
    "한": ["HAN"],
    "홍": ["HONG"],
    "용": ["YONG", "RYONG", "LONG", "RONG"],
    "명": ["MYUNG", "MYEONG", "MYONG", "MING"],
    "평": ["PYUNG", "PYEONG", "PHYONG", "PYONG", "PING"],
    "령": ["RYONG", "RYEONG", "RYUNG", "LYONG", "LING"],
    "룡": ["RYONG", "LONG", "YONG"],
    "천": ["CHUN", "CHEON", "CHON", "CHEN", "TIAN"],
    "전": ["JEON", "CHON", "JUN", "JON", "ZHAN"],
    "철": ["CHUL", "CHEOL"],
    "원": ["WON", "ONE", "YUAN", "WEN"],
    "연": ["YUN", "YEON", "YAN", "LIAN"],
    "흥": ["HEUNG", "HUNG"],
    "봉": ["BONG", "PONG", "FENG"],
    "통": ["TONG"],
    "국": ["GUK", "KUK", "KOOK", "GUO"],
    "덕": ["DEOK", "DUK", "DUCK", "DOK", "DE"],
    "복": ["BOK", "POK", "FU"],
    "록": ["ROK", "LOK", "LU"],
    "락": ["RAK", "LAK", "LE"],
    "란": ["RAN", "LAN"],
    "림": ["RIM", "LIM", "LIN"],
    "립": ["RIP", "LIP"],
    "압": ["AP", "AB"],
    "해": ["HAE", "HAI", "PHAE"],
    "백": ["BAEK", "PAEK", "PAIK", "BAI"],
    "청": ["CHEONG", "CHUNG", "CHONG", "QING"],
    "옥": ["OK", "OCK", "YU"],
    "학": ["HAK", "HAG"],
    "협": ["HYUP", "HYEOP", "XIE"],
    "새": ["SAE", "SE", "SHAE"],
    "호": ["HO", "HU", "HAO"],
    "훼": ["FE", "FUE", "HWE", "HUI"],
    "지": ["JI", "ZHI", "ZI", "GI", "GE", "ZY", "G"],
    "치": ["CHI", "CHIH", "QI", "CI"],
    "산": ["SAN", "SHAN"],
    "샨": ["SHAN", "XIAN"],
    "셍": ["SHENG", "SENG"],
    "셩": ["SHENG", "XIONG"],
    "헹": ["HENG"],
    "쳉": ["CHENG"],
    "펭": ["FENG", "PENG"],
    "슈": ["XIU", "SHU", "SYU"],
    "렌": ["REN", "LEN"],
    "링": ["LING", "RING"],
    "닝": ["NING"],
    "첸": ["CHEN", "CHAN", "QIAN"],
    "다": ["DA", "TA"],
    "하": ["HA", "HAR", "XIA"],
    "오": ["O", "OH", "AO", "OR", "AU"],
    "저": ["ZHOU", "JEO", "ZHE"],
    "센": ["SHEN", "SEN", "CEN"],
    "위": ["WI", "WEI", "YU", "UI"],
    "웨": ["WE", "WEI", "WAE"],
    "퍼": ["PER", "PUR", "FUR", "PO", "PEO"],
    "버": ["BER", "BUR", "VER", "BO", "BEO"],
    "터": ["TER", "TUR", "TO", "TEO"],
    "서": ["SUR", "SER", "SEO", "SU"],
    "커": ["KER", "CUR", "COR", "KEO"],
    "머": ["MER", "MUR", "MEO"],
    "더": ["DER", "DUR", "THE", "DEO"],
    "너": ["NER", "NUR", "NEO"],
    "러": ["RUR", "LER", "RU", "LU"],
    "스": ["S", "SE", "SU", "SHI", "SI"],
    "트": ["T", "TE", "TU"],
    "드": ["D", "DE", "DU"],
    "크": ["K", "C", "KE", "CK"],
    "프": ["P", "F", "PE", "FE", "PH"],
    "브": ["B", "V", "BE", "VE"],
    "즈": ["Z", "S", "ZE", "SE", "ZHI", "ZI"],
    "츠": ["TS", "TZ", "Z", "C", "CHI", "CI"],
    "르": ["R", "L", "RE", "LE", "RI"],
    "글": ["GLE", "GL", "GEUL"],
    "블": ["BLE", "BL"],
    "플": ["PLE", "PL", "FL", "FLE"],
    "틀": ["TLE", "TL"],
    "클": ["CLE", "CL", "KL"],
    "파": ["PA", "FA", "PAR", "FAR"],
    "피": ["PI", "FI", "PEE", "FEE", "PHY", "P"],
    "페": ["PE", "FE", "PAE", "FA"],
    "포": ["PO", "FO", "FOR", "POR", "PHO"],
    "푸": ["PU", "FU", "POO", "FOO"],
    "바": ["BA", "VA", "BAR", "VAR"],
    "비": ["BI", "VI", "BEE", "VEE", "VY", "B"],
    "베": ["BE", "VE", "BAE", "VA"],
    "보": ["BO", "VO", "BOR", "VOR"],
    "라": ["RA", "LA", "RAR", "LAR"],
    "리": ["RI", "LI", "RY", "LY", "LEE", "REE"],
    "루": ["RU", "LU", "ROO", "LOO"],
    "레": ["RE", "LE", "RAE", "LAE"],
    "로": ["RO", "LO", "ROR", "LOR", "RHO"],
    "마": ["MA", "MAR"],
    "카": ["KA", "CA", "CAR", "KAR"],
    "타": ["TA", "TAR"],
    "아": ["A", "AR", "AH"],
    "코": ["KO", "CO", "COR", "KOR"],
    "노": ["NO", "NOR"],
    "도": ["DO", "DOR", "DAO"],
    "모": ["MO", "MOR"],
    "소": ["SO", "SOR"],
    "토": ["TO", "TOR"],
    "싱": ["XING", "SING", "SHING"],
    "샤": ["XIA", "SHA", "SHIA"],
    "샹": ["XIANG", "SHANG", "CHANG"],
    "항": ["HANG", "KANG"],
    "화": ["HUA", "WHA", "HWA"],
    "중": ["ZHONG", "JUNG", "CHUNG"],
    "장": ["JIANG", "ZHANG", "JANG", "CHANG"],
    "칭": ["QING", "CHING"],
    "둥": ["DONG"],
    "난": ["NAN"],
    "순": ["SHUN", "SOON", "SUN"],
    "룬": ["LUN", "RUN"],
    "륭": ["LONG", "RYUNG"],
    "룽": ["LONG", "RUNG"],
    "펑": ["FENG", "PENG", "PUNG"],
    "궈": ["GUO", "KWO"],
    "밍": ["MING"],
    "양": ["YANG"],
    "류": ["RYU", "LIU"],
    "쇼": ["SHO", "SYO"],
    "요": ["YO", "YAO"],
    "투": ["TWO", "TU", "TOO", "TUO"],
    "시": ["SI", "SHI", "SEA", "C"],
    "씨": ["C", "SEA", "SSI"],
    "디": ["DI", "D", "DE"],
    "티": ["TI", "T", "TEA"],
    "앤": ["N", "AN", "AND", "AEN"],
    "엔": ["N", "EN"],
    "엠": ["M", "EM"],
    "엘": ["L", "EL"],
    "누": ["NU", "NOO"],
    "표": ["PYO", "PHYO"],
}

IDEA_B_WORD_VARS: Dict[str, List[str]] = {
    "마루": ["MARU"], "위안": ["YUAN"], "유안": ["YUAN"], "루이": ["RUI"], "후이": ["HUI"],
    "투오": ["TUO"], "구오": ["GUO"], "루오": ["LUO"], "샤오": ["XIAO", "SHAO"], "자오": ["ZHAO", "JIAO"],
    "차오": ["CHAO", "QIAO"], "저우": ["ZHOU"], "베이": ["BEI", "BAY"], "하이": ["HAI", "HIGH", "HI"], "타이": ["TAI", "THAI"],
    "바오": ["BAO"], "다오": ["DAO"], "하오": ["HAO"], "다롄": ["DALIAN"], "칭다오": ["QINGDAO"],
    "상하이": ["SHANGHAI"], "닝보": ["NINGBO"], "텐진": ["TIANJIN"], "선전": ["SHENZHEN"], "광저우": ["GUANGZHOU"],
    "샤먼": ["XIAMEN"], "홍콩": ["HONGKONG"], "후지": ["FUJI"], "사쿠라": ["SAKURA"], "히마와리": ["HIMAWARI"],
    "스타": ["STAR"], "오션": ["OCEAN"], "글로리": ["GLORY"], "피오니아": ["PIONEER"], "파이오니아": ["PIONEER"],
    "프린스": ["PRINCE"], "프린세스": ["PRINCESS"], "브릿지": ["BRIDGE"], "브리지": ["BRIDGE"], "스피릿": ["SPIRIT"],
    "프라이드": ["PRIDE"], "트레이더": ["TRADER"], "익스프레스": ["EXPRESS"], "엔터프라이즈": ["ENTERPRISE"], "챌린저": ["CHALLENGER"],
    "보이저": ["VOYAGER"], "프로그레스": ["PROGRESS"], "하모니": ["HARMONY"], "빅토리": ["VICTORY"], "유니온": ["UNION"],
    "유니언": ["UNION"], "리버티": ["LIBERTY"], "트라이엄프": ["TRIUMPH"], "센추리": ["CENTURY"], "퍼시픽": ["PACIFIC"],
    "아틀란틱": ["ATLANTIC"], "그린": ["GREEN"], "블루": ["BLUE"], "골든": ["GOLDEN"], "실버": ["SILVER"],
    "다이아몬드": ["DIAMOND"], "크리스탈": ["CRYSTAL"], "썬라이즈": ["SUNRISE"], "선라이즈": ["SUNRISE"], "썬셋": ["SUNSET"],
    "선셋": ["SUNSET"], "레인보우": ["RAINBOW"], "에메랄드": ["EMERALD"], "사파이어": ["SAPPHIRE"], "주피터": ["JUPITER"],
    "비너스": ["VENUS"], "머큐리": ["MERCURY"], "넵튠": ["NEPTUNE"], "아폴로": ["APOLLO"], "아테나": ["ATHENA"],
    "포세이돈": ["POSEIDON"], "제우스": ["ZEUS"], "페가수스": ["PEGASUS"], "오리온": ["ORION"], "시리우스": ["SIRIUS"],
    "폴라리스": ["POLARIS"], "노스": ["NORTH"], "사우스": ["SOUTH"], "이스트": ["EAST"], "웨스트": ["WEST"],
    "케미": ["CHEMI", "CHEM"], "케미칼": ["CHEMICAL"], "탱커": ["TANKER"], "마린": ["MARINE"], "마리타임": ["MARITIME"],
    "내비게이터": ["NAVIGATOR"], "네비게이터": ["NAVIGATOR"], "리더": ["LEADER", "LIDER"], "에이스": ["ACE"], "챔피언": ["CHAMPION"],
    "프론티어": ["FRONTIER"], "호라이즌": ["HORIZON"], "드래곤": ["DRAGON"], "피닉스": ["PHOENIX"], "이글": ["EAGLE"],
    "팔콘": ["FALCON"], "스완": ["SWAN"], "알바트로스": ["ALBATROSS"],
    "훼리": ["FERRY"], "페리": ["FERRY"], "카훼리": ["CARFERRY"], "카페리": ["CARFERRY"], "고속": ["KOSOK", "GOSOK", "EXPRESS", "FAST"],
    "크루즈": ["CRUISE"], "크루저": ["CRUISER"], "썬크루저": ["SUNCRUISER"], "선크루저": ["SUNCRUISER"], "퀸": ["QUEEN"],
    "킹": ["KING"], "아일랜드": ["ISLAND"], "아일랜더": ["ISLANDER"], "플라워": ["FLOWER"], "씨플라워": ["SEAFLOWER"],
    "클라우드": ["CLOUD"], "실버클라우드": ["SILVERCLOUD"], "드림": ["DREAM"], "판스타": ["PANSTAR"], "팬스타": ["PANSTAR"],
    "씨스타": ["SEASTAR"], "시스타": ["SEASTAR"], "도라도": ["DORADO"], "엘도라도": ["ELDORADO"], "파라다이스": ["PARADISE"],
    "마리너": ["MARINER"], "셔틀": ["SHUTTLE"], "젯": ["JET"], "제트": ["JET", "Z"], "하이제트": ["HIJET"],
    "씨디": ["SIDDY", "CD"], "시디": ["SIDDY", "CD"], "신디": ["SIDDY", "CINDY"], "돌핀": ["DOLPHIN"], "카니발": ["CARNIVAL"],
    "그란데": ["GRANDE"], "트로픽": ["TROPIC"], "캐리어": ["CARRIER"], "시웨이즈": ["SEAWAYS"], "씨웨이즈": ["SEAWAYS"],
    "시즈": ["SEAS"], "씨즈": ["SEAS"], "캡틴": ["CAPTAIN", "KAPTAN"], "카프탄": ["KAPTAN", "CAPTAIN"], "캅탄": ["KAPTAN", "CAPTAIN"],
    "벨루가": ["BELUGA"], "아이리스": ["IRIS"], "카멜리아": ["CAMELLIA"], "에코누리": ["ECONURI"], "누리": ["NURI"],
    "한우리": ["HANWOORI", "HANURI"], "우리": ["WOORI", "URI"], "코모도": ["COMMODORE", "KOMODO"], "코모도어": ["COMMODORE"],
    "쓰시마": ["TSUSHIMA"], "대마도": ["TSUSHIMA", "DAEMADO"], "링크": ["LINK"], "우바": ["UVA"], "브이아이피": ["VIP"],
    "삼표": ["SAMPYO"], "코렉스": ["KOREX"], "세인": ["SEIN"], "카사마": ["KASAMA"], "자모산": ["JAMOSAN"],
    "수양산": ["SUYANGSAN"], "칠성": ["CHILSUNG", "CHILSEONG"], "코넬리아": ["CORNELIA"], "모닝": ["MORNING"], "써니": ["SUNNY"],
    "선니": ["SUNNY"], "데이지": ["DAISY"], "델리": ["DELHI"], "시네스": ["SINES"], "유니버설": ["UNIVERSAL"],
    "유니버살": ["UNIVERSAL"], "파트너": ["PARTNER"], "태창": ["TAECHANG"], "항로": ["HANGRO"], "보성": ["BOSUNG", "BOSEONG"],
    "신해": ["SINHAE", "SHINHAE"], "해도": ["HAEDO"], "장자": ["JANGJA"], "누산타라": ["NUSANTARA"], "사북": ["SABUK"],
    "다르마": ["DHARMA"], "바하리": ["BAHARI", "BAHRI"], "바흐리": ["BAHRI", "BAHARI"], "스테나": ["STENA"], "유로카고": ["EUROCARGO"],
    "파자르": ["FAJAR"], "자발": ["JABAL"], "트랜스": ["TRANS"], "포인트": ["POINT"], "미드나잇": ["MIDNIGHT"],
    "시스팬": ["SEASPAN"], "씨스팬": ["SEASPAN"], "아크": ["ARK"], "플라잉": ["FLYING"], "패스트": ["FAST"],
    "마리아": ["MARIA"], "골드": ["GOLD"],
    "산동": ["SHANDONG", "SANDONG"], "샨동": ["SHANDONG"], "산둥": ["SHANDONG"], "웨이하이": ["WEIHAI"], "위해": ["WEIHAI"],
    "장인": ["JIANGYIN"], "장음": ["JIANGYIN"], "저우산": ["ZHOUSHAN"], "닝하이": ["NINGHAI"], "닝하이투오": ["NINGHAITUO"],
    "닝후아": ["NINGHUA"], "치셍": ["CHIHSHENG"], "딩헹": ["DINGHENG"], "지하이": ["ZYHI", "ZHIHAI", "JIHAI"], "펭슈하이": ["FENGXIUHAI"],
    "펭슈": ["FENGXIU"], "한렌": ["HANREN"], "푸루이": ["FURUI"], "푸후이": ["FUHUI"], "푸유안": ["FUYUAN"],
    "바오루이링": ["BAORUILING"], "바오루이": ["BAORUI"], "다랴": ["DARYA"], "첸드": ["CHAND"], "차담": ["CHATHAM"],
    "안딩하이": ["ANDINGHAI"], "안딩": ["ANDING"], "유센": ["YOUSHEN"], "신간": ["SINGAN", "XINGAN"],
    "압해": ["APHAE"], "욕지": ["YOKJI"], "영동": ["YEONGDONG", "YONGDONG"], "청산": ["CHEONGSAN", "CHUNGSAN"], "천사": ["CHEONSA", "ANGEL"],
    "금오": ["GUMOH", "GEUMO", "KUMOH"], "덕일": ["DUCKIL", "DEOKIL"], "두루봉": ["TURUBONG", "DURUBONG"], "광양": ["KWANGYANG", "GWANGYANG"],
    "여수": ["YEOSU", "YOSU"], "인천": ["INCHEON", "INCHON"], "부산": ["BUSAN", "PUSAN"], "울산": ["ULSAN"], "포항": ["POHANG"],
    "목포": ["MOKPO"], "군산": ["GUNSAN", "KUNSAN"], "평택": ["PYEONGTAEK", "PYONGTAEK", "PHYONGTAEK"], "동해": ["DONGHAE", "EASTSEA"],
    "통영": ["TONGYEONG", "TONGYONG"], "거제": ["GEOJE", "KOJE"], "옥포": ["OKPO"], "고현": ["GOHYEON", "KOHYON"], "마산": ["MASAN"],
    "진해": ["JINHAE", "CHINHAE"], "삼척": ["SAMCHEOK", "SAMCHOK"], "속초": ["SOKCHO"], "완도": ["WANDO"], "제주": ["JEJU", "CHEJU"],
    "은구름": ["SILVERCLOUD", "EUNGUREUM"], "해성": ["SEASTAR", "HAESUNG", "HAESEONG"], "해화": ["SEAFLOWER", "HAEHWA"], "바다꽃": ["SEAFLOWER"],
    "한국": ["KOREA", "HANKUK", "HANGUK"], "대한": ["KOREA", "DAEHAN"], "코리아": ["KOREA"], "극동": ["FAREAST", "KEUKDONG", "GUKDONG"],
    "태평양": ["PACIFIC", "TAEPYUNGYANG"], "대서양": ["ATLANTIC"], "무지개": ["RAINBOW", "MUJIGAE"], "태양": ["SUN", "TAEYANG"],
    "해오름": ["SUNRISE", "HAEOREUM"], "일출": ["SUNRISE", "ILCHUL"], "평화": ["PEACE", "PYUNGHWA", "PYEONGHWA"],
    "자유": ["LIBERTY", "FREEDOM", "JAYU"], "희망": ["HOPE", "HUIMANG"], "승리": ["VICTORY", "SEUNGRI"], "영광": ["GLORY", "YEONGKWANG"],
    "조화": ["HARMONY", "JOHWA"], "독수리": ["EAGLE", "DOKSURI"], "백조": ["SWAN", "BAEKJO"], "돌고래": ["DOLPHIN", "DOLGORAE"],
    "거북": ["TURTLE", "KOBUK", "GEOBUK"],
    "원": ["ONE", "WON"], "투": ["TWO", "TU"], "쓰리": ["THREE"], "스리": ["THREE", "SRI"], "포": ["FOUR", "PO"],
    "파이브": ["FIVE"], "식스": ["SIX"], "세븐": ["SEVEN"], "에잇": ["EIGHT"], "나인": ["NINE"], "텐": ["TEN"],
    "퍼스트": ["FIRST"], "세컨드": ["SECOND"], "써드": ["THIRD"],
}

PINYIN_BIGRAM_CONTRACTIONS: List[Tuple[str, str]] = [
    ("투오", "TUO"), ("루이", "RUI"), ("후이", "HUI"), ("유안", "YUAN"), ("위안", "YUAN"),
    ("샤오", "XIAO"), ("자오", "ZHAO"), ("차오", "CHAO"), ("바오", "BAO"), ("다오", "DAO"),
    ("하오", "HAO"), ("가오", "GAO"), ("라오", "LAO"), ("타오", "TAO"), ("저우", "ZHOU"),
    ("구오", "GUO"), ("루오", "LUO"), ("수이", "SUI"), ("추이", "CUI"), ("구이", "GUI"),
    ("웨이", "WEI"),
]

COMPOUND_SUBWORD_REPLACEMENTS: List[Tuple[str, str]] = [
    ("카훼리", "CARFERRY"), ("카페리", "CARFERRY"), ("썬크루저", "SUNCRUISER"), ("선크루저", "SUNCRUISER"),
    ("씨플라워", "SEAFLOWER"), ("실버클라우드", "SILVERCLOUD"), ("하이제트", "HIJET"), ("에코누리", "ECONURI"),
    ("한우리", "HANWOORI"), ("아일랜더", "ISLANDER"), ("아일랜드", "ISLAND"), ("익스프레스", "EXPRESS"),
    ("프린세스", "PRINCESS"), ("파라다이스", "PARADISE"), ("프론티어", "FRONTIER"), ("유로카고", "EUROCARGO"),
    ("누산타라", "NUSANTARA"), ("코모도어", "COMMODORE"), ("코모도", "COMMODORE"), ("쓰시마", "TSUSHIMA"),
    ("판스타", "PANSTAR"), ("팬스타", "PANSTAR"), ("씨스타", "SEASTAR"), ("시스타", "SEASTAR"),
    ("시웨이즈", "SEAWAYS"), ("씨웨이즈", "SEAWAYS"), ("훼리", "FERRY"), ("페리", "FERRY"),
    ("크루즈", "CRUISE"), ("크루저", "CRUISER"), ("플라워", "FLOWER"), ("드림", "DREAM"),
    ("하모니", "HARMONY"), ("글로리", "GLORY"), ("오션", "OCEAN"), ("스타", "STAR"),
    ("퀸", "QUEEN"), ("제트", "JET"), ("고속", "KOSOK"), ("은구름", "SILVERCLOUD"), ("해성", "SEASTAR"),
]

re_hangul = re.compile(r"[가-힣]")
re_non_alnum_kr = re.compile(r"[^A-Z0-9가-힣]+")


def is_korean_name(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return bool(re_hangul.search(name))


_JAMO_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JAMO_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JAMO_JONG = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
    "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
_PHONETIC_JAMO_MAP = str.maketrans({
    "ㅐ": "ㅔ", "ㅒ": "ㅖ", "ㅙ": "ㅞ", "ㅚ": "ㅞ",
    "ㄲ": "ㅋ", "ㄸ": "ㅌ", "ㅃ": "ㅍ", "ㅆ": "ㅅ", "ㅉ": "ㅈ",
})
EN_LETTER_TO_KR = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
    "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘",
    "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알",
    "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "제트",
}


def hangul_to_jamo(text: str, phonetic_norm: bool = True) -> str:
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            cho = _JAMO_CHO[offset // 588]
            jung = _JAMO_JUNG[(offset % 588) // 28]
            jong = _JAMO_JONG[offset % 28]
            out.append(cho)
            out.append(jung)
            if jong:
                if phonetic_norm:
                    if jong in ("ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㄷ"):
                        jong = "ㅅ"
                    elif jong in ("ㅋ", "ㄲ"):
                        jong = "ㄱ"
                    elif jong == "ㅍ":
                        jong = "ㅂ"
                out.append(jong)
        elif "A" <= ch <= "Z":
            kr_eq = EN_LETTER_TO_KR.get(ch, "")
            if kr_eq:
                out.append(hangul_to_jamo(kr_eq, phonetic_norm=phonetic_norm))
        elif "0" <= ch <= "9":
            out.append(ch)
    res = "".join(out)
    return res.translate(_PHONETIC_JAMO_MAP) if phonetic_norm else res


def load_default_full_ko_map(base_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    search_roots = []
    if base_dir is not None:
        search_roots.append(base_dir)
    try:
        search_roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    search_roots.append(Path.cwd())
    for root in search_roots:
        for rel in ("model/catalog_full_ko_map.json", "catalog_full_ko_map.json", "AR-06/catalog_full_ko_map.json"):
            cand = root / rel
            if cand.exists():
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
    return {}


def syllable_to_rom(ch: str) -> str:
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return ch
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return CHOSUNG[cho] + JUNGSUNG[jung] + JONGSUNG[jong]


def en_to_phonetic(word: str) -> str:
    if not word:
        return ""
    s = re.sub(r"[^A-Z0-9]", "", word.upper())
    if not s:
        return ""
    for pat, rep in [
        ("TCH", "C"), ("SCH", "S"), ("SH", "S"), ("CH", "C"),
        ("PH", "P"), ("GH", ""), ("CK", "K"), ("QU", "KW"),
        ("WR", "R"), ("KN", "N"), ("WH", "W"), ("TH", "S"),
        ("NG", "N"),
        ("OO", "U"), ("EE", "I"), ("EA", "I"), ("OA", "O"),
        ("OU", "U"), ("OW", "O"), ("AI", "E"), ("AY", "E"),
        ("AU", "O"), ("AW", "O"), ("EW", "U"), ("IE", "I"),
    ]:
        s = s.replace(pat, rep)
    if not s:
        return ""
    if len(s) >= 4 and s.endswith("E") and s[-2] not in "AEIOU":
        s = s[:-1]
    trans = str.maketrans({
        "G": "K", "C": "K", "Q": "K",
        "B": "P", "V": "P", "F": "P",
        "D": "T", "L": "R", "Z": "S", "X": "KS",
    })
    s = s.translate(trans)
    if not s:
        return ""
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def kr_to_phonetic(word: str) -> str:
    if not word:
        return ""
    raw_pieces = []
    for ch in word:
        if "가" <= ch <= "힣":
            code = ord(ch) - 0xAC00
            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            if jung == 18 and jong == 0 and CHOSUNG_PHON[cho] in ("S", "T", "P", "K", "J", "C", "R"):
                raw_pieces.append(CHOSUNG_PHON[cho])
            else:
                raw_pieces.append(CHOSUNG_PHON[cho] + JUNGSUNG_PHON[jung] + JONGSUNG_PHON[jong])
        else:
            raw_pieces.append(ch)
    s = "".join(raw_pieces)
    if not s:
        return ""
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def consonant_skeleton(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not text:
        return ""
    text = text.replace("PH", "P").replace("SH", "S").replace("CH", "C").replace("CK", "K").replace("NG", "N")
    trans = str.maketrans({
        "P": "P", "F": "P", "V": "P", "B": "P",
        "L": "R", "C": "K", "Q": "K", "G": "K",
        "Z": "S", "D": "T",
    })
    t = text.translate(trans)
    out = []
    for i, ch in enumerate(t):
        if ch in "AEIOUWY":
            if i == 0:
                out.append("V")
            continue
        if not out or ch != out[-1]:
            out.append(ch)
    return "".join(out)


def extract_kr_acronym_prefix(word: str) -> Optional[Tuple[str, str]]:
    if len(word) < 2 or not re_hangul.search(word):
        return None
    pos = 0
    letters: List[str] = []
    n = len(word)
    while pos < n:
        matched = False
        for kr_let, en_let in KR_LETTER_ITEMS_SORTED:
            if word.startswith(kr_let, pos):
                if len(kr_let) == 1 and kr_let in ("이", "오", "유", "앤", "앰", "앨") and len(letters) == 0:
                    continue
                letters.append(en_let)
                pos += len(kr_let)
                matched = True
                break
        if not matched:
            break
    if len(letters) >= 2 and (pos >= 3 or pos == n):
        return "".join(letters), word[pos:]
    return None


def is_en_inner_typo(w1: str, w2: str) -> int:
    if len(w1) < 6 or len(w2) < 6:
        return 0
    if w1[:2] != w2[:2] or w1[-2:] != w2[-2:]:
        return 0
    if abs(len(w1) - len(w2)) > 2:
        return 0
    d = OSA.distance(w1, w2)
    if d == 1:
        return 1
    if d == 2 and len(w1) >= 7 and len(w2) >= 7:
        return 2
    return 0


def normalize_vessel_name(raw_name: Any, kr_mode: str = "AR06") -> Tuple[str, List[str], List[str]]:
    if not isinstance(raw_name, str):
        return "", [], []
    s = raw_name.strip().upper()
    for k, v in ROMAN_CHAR_MAP.items():
        s = s.replace(k, f" {v} ")

    s = re.sub(r"\bNO\.?\s*(\d+)\b", r" \1 ", s)
    s = re.sub(r"#\s*(\d+)\b", r" \1 ", s)
    s = re.sub(r"제\s*(\d+)\s*호", r" \1 ", s)
    if kr_mode in ("AR05", "AR06"):
        s = re.sub(r"제\s*(\d+)", r" \1 ", s)
    s = re.sub(r"(\d+)\s*호\b", r" \1 ", s)

    s = re.sub(r"([A-Z가-힣])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Z가-힣])", r"\1 \2", s)
    s = re.sub(r"([A-Z])([가-힣])", r"\1 \2", s)
    s = re.sub(r"([가-힣])([A-Z])", r"\1 \2", s)

    raw_tokens = [tok for tok in re_non_alnum_kr.split(s) if tok]
    word_tokens: List[str] = []
    num_tokens: List[str] = []

    for i, tok in enumerate(raw_tokens):
        if tok in ROMAN_TOKEN_MAP and (i == len(raw_tokens) - 1 or len(raw_tokens) > 1):
            num_tokens.append(str(int(ROMAN_TOKEN_MAP[tok])))
            continue
        if tok.isdigit():
            num_tokens.append(str(int(tok)))
            continue
        if tok == "NO":
            continue
        if tok == "호":
            continue
        if i == len(raw_tokens) - 1 and len(tok) >= 2 and tok.endswith("호") and re_hangul.search(tok):
            tok = tok[:-1]
        elif len(tok) >= 3 and tok.endswith("호") and re_hangul.search(tok):
            tok = tok[:-1]

        if kr_mode not in ("AR05", "AR06"):
            old_map = {**KR_EN_NUM_WORD_MAP, "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9", "십": "10"}
            if tok in old_map and i == len(raw_tokens) - 1 and len(raw_tokens) >= 2:
                num_tokens.append(old_map[tok])
                continue

        if tok:
            word_tokens.append(tok)

    merged_words: List[str] = []
    idx = 0
    letter_map_to_use = KR_LETTER_MAP if kr_mode in ("AR05", "AR06") else KR_LETTER_MAP_BASE
    while idx < len(word_tokens):
        if idx + 1 < len(word_tokens) and (word_tokens[idx] + word_tokens[idx + 1]) in letter_map_to_use:
            merged_words.append(letter_map_to_use[word_tokens[idx] + word_tokens[idx + 1]])
            idx += 2
        elif word_tokens[idx] in KR_LETTER_MAP_BASE and (kr_mode not in ("AR05", "AR06") or len(word_tokens[idx]) >= 2):
            merged_words.append(KR_LETTER_MAP_BASE[word_tokens[idx]])
            idx += 1
        elif kr_mode in ("AR05", "AR06") and len(word_tokens[idx]) >= 2 and word_tokens[idx] in letter_map_to_use:
            merged_words.append(letter_map_to_use[word_tokens[idx]])
            idx += 1
        else:
            merged_words.append(word_tokens[idx])
            idx += 1

    compact = "".join(merged_words) + ("".join(num_tokens) if num_tokens else "")
    return compact, merged_words, num_tokens


def parse_kr_row_features(raw_name: Any, kr_mode: str = "AR06") -> Tuple[str, List[str], List[str], str, bool]:
    if not isinstance(raw_name, str):
        return "", [], [], "", False

    s_raw = raw_name.strip().upper()
    raw_check_toks = [tok for tok in re_non_alnum_kr.split(s_raw) if tok]
    cond_a_2char_ho = False
    if raw_check_toks:
        last_raw = raw_check_toks[-1]
        if len(last_raw) == 2 and last_raw.endswith("호") and re_hangul.search(last_raw[0]) and not last_raw[0].isdigit():
            cond_a_2char_ho = True

    comp, wtoks, ntoks = normalize_vessel_name(raw_name, kr_mode=kr_mode)

    kr_ord_nums: List[str] = []
    if kr_mode in ("AR05", "AR06") and len(wtoks) >= 2 and wtoks[-1] in KR_EN_NUM_WORD_MAP:
        kr_ord_nums.append(KR_EN_NUM_WORD_MAP[wtoks[-1]])

    return comp, wtoks, ntoks, "|".join(kr_ord_nums), cond_a_2char_ho


def dp_align_kr_en(kr_word: str, en_word: str) -> List[Tuple[str, str]]:
    n = len(kr_word)
    m = len(en_word)
    if n == 0 or m == 0 or m < n:
        return []
    if n == 1:
        return [(kr_word, en_word)]

    syl_roms = [syllable_to_rom(ch) for ch in kr_word]
    INF = 1e9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        s_rom = syl_roms[i - 1]
        for j in range(i, m - (n - i) + 1):
            for k in range(max(i - 1, j - 5), j):
                if dp[i - 1][k] >= INF:
                    continue
                chunk = en_word[k:j]
                sim = OSA.normalized_similarity(s_rom, chunk)
                cost = (1.0 - sim) + 0.08 * abs(len(chunk) - len(s_rom))
                if dp[i - 1][k] + cost < dp[i][j]:
                    dp[i][j] = dp[i - 1][k] + cost
                    back[i][j] = k

    if dp[n][m] >= INF:
        return []

    pairs: List[Tuple[str, str]] = []
    curr_j = m
    for i in range(n, 0, -1):
        prev_k = back[i][curr_j]
        pairs.append((kr_word[i - 1], en_word[prev_k:curr_j]))
        curr_j = prev_k
    pairs.reverse()
    return pairs


def kr_word_to_en_candidates(
    word: str,
    syl_top2: Dict[str, List[str]],
    word_map: Dict[str, str],
) -> Tuple[str, str]:
    if not re_hangul.search(word):
        return word, word
    if word in word_map:
        mapped = word_map[word]
        return mapped, mapped
    if word in KR_LETTER_MAP:
        mapped = KR_LETTER_MAP[word]
        return mapped, mapped
    if len(word) >= 3:
        acr = extract_kr_acronym_prefix(word)
        if acr is not None and not acr[1]:
            return acr[0], acr[0]

    out1: List[str] = []
    out2: List[str] = []
    for ch in word:
        if "가" <= ch <= "힣":
            cands = syl_top2.get(ch, [])
            rom = syllable_to_rom(ch)
            out1.append(cands[0] if cands else rom)
            out2.append(cands[1] if len(cands) > 1 else rom)
        else:
            out1.append(ch)
            out2.append(ch)
    return "".join(out1), "".join(out2)


def generate_idea_b_word_variants(
    word: str,
    syl_top3: Dict[str, List[str]],
    word_top2: Dict[str, List[str]],
) -> List[str]:
    if not re_hangul.search(word):
        return [word]
    variants: List[str] = []

    def add_v(v: str) -> None:
        if v and v not in variants:
            variants.append(v)

    if word in word_top2:
        for v in word_top2[word]:
            add_v(v)

    if word in IDEA_B_WORD_VARS:
        for v in IDEA_B_WORD_VARS[word]:
            add_v(v)

    if word in KR_LETTER_MAP:
        add_v(KR_LETTER_MAP[word])

    acr = extract_kr_acronym_prefix(word)
    if acr is not None:
        acr_en, rem_kr = acr
        if not rem_kr:
            add_v(acr_en)
        else:
            rem_vars = generate_idea_b_word_variants(rem_kr, syl_top3, word_top2)
            for rv in rem_vars[:2]:
                add_v(acr_en + rv)

    seq_learn1: List[str] = []
    seq_learn2: List[str] = []
    seq_rom: List[str] = []
    seq_b_var1: List[str] = []
    seq_b_var2: List[str] = []
    seq_b_var3: List[str] = []

    for idx, ch in enumerate(word):
        if "가" <= ch <= "힣":
            l_cands = syl_top3.get(ch, [])
            b_cands = IDEA_B_SYLLABLE_VARS.get(ch, [])
            rom = syllable_to_rom(ch)

            c1 = l_cands[0] if l_cands else (b_cands[0] if b_cands else rom)
            c2 = l_cands[1] if len(l_cands) > 1 else (b_cands[0] if b_cands else rom)
            bv1 = b_cands[0] if b_cands else c1
            bv2 = b_cands[1] if len(b_cands) > 1 else (b_cands[0] if b_cands else rom)
            bv3 = b_cands[2] if len(b_cands) > 2 else bv2

            code = ord(ch) - 0xAC00
            jung = (code % 588) // 28
            jong = code % 28
            if jung == 18 and jong == 0 and idx == len(word) - 1 and ch in IDEA_B_SYLLABLE_VARS:
                bv1 = IDEA_B_SYLLABLE_VARS[ch][0]

            seq_learn1.append(c1)
            seq_learn2.append(c2)
            seq_rom.append(rom)
            seq_b_var1.append(bv1)
            seq_b_var2.append(bv2)
            seq_b_var3.append(bv3)
        else:
            for lst in (seq_learn1, seq_learn2, seq_rom, seq_b_var1, seq_b_var2, seq_b_var3):
                lst.append(ch)

    add_v("".join(seq_learn1))
    add_v("".join(seq_b_var1))
    add_v("".join(seq_learn2))
    add_v("".join(seq_b_var2))
    add_v("".join(seq_b_var3))
    add_v("".join(seq_rom))

    if len(word) >= 2:
        w_py = word
        replaced_py = False
        for kr_bi, en_bi in PINYIN_BIGRAM_CONTRACTIONS:
            if kr_bi in w_py:
                w_py = w_py.replace(kr_bi, f" {en_bi} ")
                replaced_py = True
        if replaced_py:
            py_pieces: List[str] = []
            for tok in w_py.split():
                if not re_hangul.search(tok):
                    py_pieces.append(tok)
                elif tok in IDEA_B_WORD_VARS:
                    py_pieces.append(IDEA_B_WORD_VARS[tok][0])
                else:
                    for ch in tok:
                        if "가" <= ch <= "힣":
                            bc = IDEA_B_SYLLABLE_VARS.get(ch, [])
                            lc = syl_top3.get(ch, [])
                            py_pieces.append(bc[0] if bc else (lc[0] if lc else syllable_to_rom(ch)))
                        else:
                            py_pieces.append(ch)
            add_v("".join(py_pieces))

    if len(word) >= 3:
        for sub_kr, sub_en in COMPOUND_SUBWORD_REPLACEMENTS:
            if sub_kr in word and word != sub_kr:
                parts = word.split(sub_kr)
                conv_parts = []
                for p in parts:
                    if not p:
                        conv_parts.append("")
                    elif p in IDEA_B_WORD_VARS:
                        conv_parts.append(IDEA_B_WORD_VARS[p][0])
                    elif p in word_top2 and word_top2[p]:
                        conv_parts.append(word_top2[p][0])
                    else:
                        p_cands = []
                        for ch in p:
                            if "가" <= ch <= "힣":
                                lc = syl_top3.get(ch, [])
                                bc = IDEA_B_SYLLABLE_VARS.get(ch, [])
                                p_cands.append(lc[0] if lc else (bc[0] if bc else syllable_to_rom(ch)))
                            else:
                                p_cands.append(ch)
                        conv_parts.append("".join(p_cands))
                add_v(sub_en.join(conv_parts))
                break

    return variants[:10]


def lattice_word_similarity(
    kr_word: str,
    en_word: str,
    syl_top3: Dict[str, List[str]],
) -> float:
    if not kr_word or not en_word or not re_hangul.search(kr_word):
        return 0.0
    n = len(kr_word)
    m = len(en_word)
    if m == 0:
        return 0.0

    syl_cand_sets: List[List[str]] = []
    for ch in kr_word:
        if "가" <= ch <= "힣":
            s_set: List[str] = []
            for x in syl_top3.get(ch, []):
                if x and x not in s_set:
                    s_set.append(x)
            for x in IDEA_B_SYLLABLE_VARS.get(ch, []):
                if x and x not in s_set:
                    s_set.append(x)
            rom = syllable_to_rom(ch)
            if rom and rom not in s_set:
                s_set.append(rom)
            syl_cand_sets.append(s_set[:8])
        else:
            syl_cand_sets.append([ch])

    NEG_INF = -1e9
    dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        cands = syl_cand_sets[i - 1]
        for j in range(1, m + 1):
            best_prev = NEG_INF
            for k in range(max(0, j - 6), j):
                if dp[i - 1][k] <= NEG_INF / 2:
                    continue
                chunk = en_word[k:j]
                best_c_sim = max(
                    1.0 if sc == chunk else OSA.normalized_similarity(sc, chunk)
                    for sc in cands
                )
                if dp[i - 1][k] + best_c_sim > best_prev:
                    best_prev = dp[i - 1][k] + best_c_sim
            dp[i][j] = best_prev

    if dp[n][m] <= NEG_INF / 2:
        return 0.0
    return float(dp[n][m] / n)


def year_bucket(year_val: Any) -> str:
    try:
        y = int(float(year_val))
    except (ValueError, TypeError):
        return "UNK"
    if y <= 2018:
        return "OLD"
    if y <= 2022:
        return "MID"
    return "NEW"


def fit_matching_model(
    train_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    kr_mode: str = "AR06",
    en_mode: str = "EN_V3",
    use_train_lookup: bool = True,
) -> Dict[str, Any]:
    merged = train_df.merge(
        catalog_df[["vessel_id", "vessel_name", "vessel_type", "flag_name", "gross_tonnage"]],
        left_on="target_vessel_id",
        right_on="vessel_id",
        suffixes=("_tr", "_cat"),
        how="inner",
    )

    callsign_counts: Dict[str, Counter] = defaultdict(Counter)
    name_counts: Dict[str, Counter] = defaultdict(Counter)

    for row in train_df.itertuples(index=False):
        vid = str(row.target_vessel_id)
        cs = str(row.call_sign).strip().upper() if pd.notna(row.call_sign) else ""
        if cs and cs not in ("NAN", "NONE", ""):
            callsign_counts[cs][vid] += 1
        compact, _, _ = normalize_vessel_name(row.vessel_name, kr_mode=kr_mode)
        if compact:
            name_counts[compact][vid] += 1

    callsign_lookup = {k: v.most_common(1)[0][0] for k, v in callsign_counts.items()}
    name_lookup = {k: v.most_common(1)[0][0] for k, v in name_counts.items()}
    unique_callsign_lookup = {k: next(iter(v)) for k, v in callsign_counts.items() if len(v) == 1}
    unique_name_lookup = {k: next(iter(v)) for k, v in name_counts.items() if len(v) == 1}

    word_pair_counts: Dict[str, Counter] = defaultdict(Counter)
    syl_chunk_counts: Dict[str, Counter] = defaultdict(Counter)

    for row in merged.itertuples(index=False):
        tr_name = row.vessel_name_tr
        cat_name = row.vessel_name_cat
        if not is_korean_name(tr_name):
            continue
        _, kr_words, _ = normalize_vessel_name(tr_name, kr_mode="AR02")
        _, en_words, _ = normalize_vessel_name(cat_name, kr_mode="AR02")
        if not kr_words or not en_words:
            continue

        if len(kr_words) == len(en_words):
            pairs = list(zip(kr_words, en_words))
        else:
            pairs = [("".join(kr_words), "".join(en_words))]

        for kw, ew in pairs:
            if not re_hangul.search(kw) or not ew.isalpha():
                continue
            word_pair_counts[kw][ew] += 1
            for s_ch, e_chunk in dp_align_kr_en(kw, ew):
                if "가" <= s_ch <= "힣" and e_chunk:
                    syl_chunk_counts[s_ch][e_chunk] += 1

    learned_word_map = {
        k: v.most_common(1)[0][0]
        for k, v in word_pair_counts.items()
        if v.most_common(1)[0][1] >= 1
    }
    learned_word_top2 = {
        k: [w for w, f in v.most_common(2) if f >= 1]
        for k, v in word_pair_counts.items()
    }
    learned_syl_top2: Dict[str, List[str]] = {}
    learned_syl_top3: Dict[str, List[str]] = {}
    for ch, cnt in syl_chunk_counts.items():
        top2 = [chunk for chunk, freq in cnt.most_common(2) if freq >= 1]
        top3 = [chunk for chunk, freq in cnt.most_common(3) if freq >= 1]
        if top2:
            learned_syl_top2[ch] = top2
        if top3:
            learned_syl_top3[ch] = top3

    type_joint: Dict[str, Counter] = defaultdict(Counter)
    cat_type_prior: Counter = Counter(catalog_df["vessel_type"].fillna("UNK").astype(str))
    total_cat = max(1, len(catalog_df))

    for row in merged.itertuples(index=False):
        kt = str(row.vessel_type_tr).strip() if pd.notna(row.vessel_type_tr) else "UNK"
        ct = str(row.vessel_type_cat).strip() if pd.notna(row.vessel_type_cat) else "UNK"
        type_joint[kt][ct] += 1

    type_log_lift: Dict[str, Dict[str, float]] = {}
    for kt, cnts in type_joint.items():
        tot_kt = sum(cnts.values())
        type_log_lift[kt] = {}
        for ct, c in cnts.items():
            p_cond = (c + 0.5) / (tot_kt + 5.0)
            p_prior = max(1e-5, cat_type_prior.get(ct, 1) / total_cat)
            type_log_lift[kt][ct] = round(float(np.clip(math.log(p_cond / p_prior), -2.5, 4.0)), 4)

    flag_joint: Dict[str, Counter] = defaultdict(Counter)
    cat_flag_prior: Counter = Counter(catalog_df["flag_name"].fillna("UNK").astype(str))

    for row in merged.itertuples(index=False):
        kf = str(row.flag_name_tr).strip() if pd.notna(row.flag_name_tr) else "UNK"
        cf = str(row.flag_name_cat).strip() if pd.notna(row.flag_name_cat) else "UNK"
        yb = year_bucket(row.arrival_year)
        if kf != "UNK":
            flag_joint[f"{kf}|{yb}"][cf] += 1
            flag_joint[f"{kf}|ALL"][cf] += 1

    flag_log_lift: Dict[str, Dict[str, float]] = {}
    for key, cnts in flag_joint.items():
        tot_k = sum(cnts.values())
        flag_log_lift[key] = {}
        for cf, c in cnts.items():
            p_cond = (c + 0.5) / (tot_k + 5.0)
            p_prior = max(1e-5, cat_flag_prior.get(cf, 1) / total_cat)
            flag_log_lift[key][cf] = round(float(np.clip(math.log(p_cond / p_prior), -2.5, 4.5)), 4)

    type_trans_prob: Dict[str, Dict[str, float]] = {}
    for kt, cnts in type_joint.items():
        tot_kt = sum(cnts.values())
        type_trans_prob[kt] = {ct: round(c / tot_kt, 6) for ct, c in cnts.items()}

    flag_trans_prob: Dict[str, Dict[str, float]] = {}
    for key, cnts in flag_joint.items():
        tot_k = sum(cnts.values())
        flag_trans_prob[key] = {cf: round(c / tot_k, 6) for cf, c in cnts.items()}

    return {
        "version": "1022-AR06",
        "kr_mode": kr_mode,
        "en_mode": en_mode,
        "use_train_lookup_default": use_train_lookup,
        "callsign_lookup": callsign_lookup,
        "name_lookup": name_lookup,
        "unique_callsign_lookup": unique_callsign_lookup,
        "unique_name_lookup": unique_name_lookup,
        "learned_word_map": learned_word_map,
        "learned_word_top2": learned_word_top2,
        "learned_syl_top2": learned_syl_top2,
        "learned_syl_top3": learned_syl_top3,
        "type_log_lift": type_log_lift,
        "flag_log_lift": flag_log_lift,
        "type_trans_prob": type_trans_prob,
        "flag_trans_prob": flag_trans_prob,
    }


class CatalogIndex:
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        full_ko_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.vessel_ids: List[str] = catalog_df["vessel_id"].astype(str).tolist()
        self.n = len(self.vessel_ids)
        self.vid_to_idx: Dict[str, int] = {vid: i for i, vid in enumerate(self.vessel_ids)}
        raw_ko_map = full_ko_map if full_ko_map is not None else load_default_full_ko_map()
        self.full_ko_map: Dict[str, List[str]] = {}
        for k_name, v_list in raw_ko_map.items():
            if not v_list:
                continue
            norm_k = " ".join(str(k_name).strip().upper().split())
            self.full_ko_map[norm_k] = v_list
            comp_k, w_k, _ = normalize_vessel_name(norm_k, kr_mode="AR02")
            if comp_k and comp_k not in self.full_ko_map:
                self.full_ko_map[comp_k] = v_list
            alpha_k = "".join(w_k)
            if alpha_k and alpha_k not in self.full_ko_map:
                self.full_ko_map[alpha_k] = v_list

        raw_names = catalog_df["vessel_name"].fillna("").astype(str).tolist()
        self.types = catalog_df["vessel_type"].fillna("UNK").astype(str).str.strip().tolist()
        self.flags = catalog_df["flag_name"].fillna("UNK").astype(str).str.strip().tolist()
        self.gt = pd.to_numeric(catalog_df["gross_tonnage"], errors="coerce").fillna(-1.0).to_numpy(dtype=np.float64)

        self.compact_names: List[str] = [""] * self.n
        self.word_Alpha: List[str] = [""] * self.n
        self.sorted_word_Alpha: List[str] = [""] * self.n
        self.word_tokens: List[List[str]] = [[] for _ in range(self.n)]
        self.sorted_word_tokens: List[List[str]] = [[] for _ in range(self.n)]
        self.phon_words: List[List[str]] = [[] for _ in range(self.n)]
        self.phon_full: List[str] = [""] * self.n
        self.num_strs: List[str] = [""] * self.n
        self.num_strs_ord: List[str] = [""] * self.n
        self.skeletons: List[str] = [""] * self.n

        self.exact_compact_idx: Dict[str, List[int]] = defaultdict(list)
        self.exact_alpha_idx: Dict[str, List[int]] = defaultdict(list)
        self.exact_sorted_alpha_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix3_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix2_idx: Dict[str, List[int]] = defaultdict(list)
        self.suffix3_idx: Dict[str, List[int]] = defaultdict(list)
        self.skel3_idx: Dict[str, List[int]] = defaultdict(list)
        self.skel2_idx: Dict[str, List[int]] = defaultdict(list)
        self.phon3_idx: Dict[str, List[int]] = defaultdict(list)
        self.num_idx: Dict[str, List[int]] = defaultdict(list)
        self.word_idx: Dict[str, List[int]] = defaultdict(list)
        self.outer2_idx: Dict[str, List[int]] = defaultdict(list)

        self.synth_jamo_raw: List[Tuple[str, ...]] = [() for _ in range(self.n)]
        self.synth_jamo_norm: List[Tuple[str, ...]] = [() for _ in range(self.n)]
        self.synth_jamo_wtoks: List[Tuple[List[str], ...]] = [() for _ in range(self.n)]
        self.exact_jamo_idx: Dict[str, List[int]] = defaultdict(list)
        self.jamo_prefix3_idx: Dict[str, List[int]] = defaultdict(list)
        self.jamo_suffix3_idx: Dict[str, List[int]] = defaultdict(list)
        self.jamo_word_idx: Dict[str, List[int]] = defaultdict(list)
        self.jamo_ngram3_idx: Dict[str, List[int]] = defaultdict(list)
        self.jamo_first_char_idx: Dict[str, List[int]] = defaultdict(list)

        for i, rname in enumerate(raw_names):
            comp, wtoks, ntoks = normalize_vessel_name(rname, kr_mode="AR02")
            alpha = "".join(wtoks)
            s_wtoks = sorted(wtoks)
            s_alpha = "".join(s_wtoks)
            nstr = "|".join(ntoks)
            skel = consonant_skeleton(alpha)
            p_words = [en_to_phonetic(w) for w in wtoks]
            p_full = "".join(p_words)

            ord_nums = list(ntoks)
            if wtoks and wtoks[-1] in EN_NUM_WORD_MAP:
                ord_nums.append(EN_NUM_WORD_MAP[wtoks[-1]])

            self.compact_names[i] = comp
            self.word_Alpha[i] = alpha
            self.sorted_word_Alpha[i] = s_alpha
            self.word_tokens[i] = wtoks
            self.sorted_word_tokens[i] = s_wtoks
            self.phon_words[i] = p_words
            self.phon_full[i] = p_full
            self.num_strs[i] = nstr
            self.num_strs_ord[i] = "|".join(ord_nums)
            self.skeletons[i] = skel

            if comp:
                self.exact_compact_idx[comp].append(i)
            if alpha:
                self.exact_alpha_idx[alpha].append(i)
                self.prefix3_idx[alpha[:3]].append(i)
                self.prefix2_idx[alpha[:2]].append(i)
                if len(alpha) >= 4:
                    self.suffix3_idx[alpha[-3:]].append(i)
            if s_alpha:
                self.exact_sorted_alpha_idx[s_alpha].append(i)
            if skel:
                self.skel3_idx[skel[:3]].append(i)
                self.skel2_idx[skel[:2]].append(i)
            if p_full:
                self.phon3_idx[p_full[:3]].append(i)
            if nstr:
                self.num_idx[nstr].append(i)
            for w in wtoks:
                if len(w) >= 3:
                    self.word_idx[w[:4]].append(i)
                if len(w) >= 5:
                    self.outer2_idx[f"{w[:2]}_{w[-2:]}"].append(i)

            norm_rname = " ".join(rname.strip().upper().split())
            ko_cands = (
                self.full_ko_map.get(norm_rname)
                or self.full_ko_map.get(comp)
                or self.full_ko_map.get(alpha)
                or []
            )
            if ko_cands:
                j_raw_list: List[str] = []
                j_norm_list: List[str] = []
                j_wtoks_list: List[List[str]] = []
                seen_ng3, seen_first, seen_ex, seen_p3, seen_s3, seen_jw = set(), set(), set(), set(), set(), set()
                for ko_str in ko_cands:
                    _, kw_toks, _ = normalize_vessel_name(ko_str, kr_mode="AR06")
                    if not kw_toks:
                        continue
                    jr = "".join(hangul_to_jamo(w, phonetic_norm=False) for w in kw_toks if w)
                    jw_list = [hangul_to_jamo(w, phonetic_norm=True) for w in kw_toks if w]
                    jn = "".join(jw_list)
                    if not jn:
                        continue
                    j_raw_list.append(jr)
                    j_norm_list.append(jn)
                    j_wtoks_list.append(jw_list)
                    if jn not in seen_ex:
                        seen_ex.add(jn)
                        self.exact_jamo_idx[jn].append(i)
                    if len(jn) >= 3 and jn[:3] not in seen_p3:
                        seen_p3.add(jn[:3])
                        self.jamo_prefix3_idx[jn[:3]].append(i)
                    if len(jn) >= 4 and jn[-3:] not in seen_s3:
                        seen_s3.add(jn[-3:])
                        self.jamo_suffix3_idx[jn[-3:]].append(i)
                    if jn[0] not in seen_first:
                        seen_first.add(jn[0])
                        self.jamo_first_char_idx[jn[0]].append(i)
                    for jw in jw_list:
                        if len(jw) >= 3 and jw[:4] not in seen_jw:
                            seen_jw.add(jw[:4])
                            self.jamo_word_idx[jw[:4]].append(i)
                    pad_jn = f" {jn} "
                    for ng_i in range(max(0, len(pad_jn) - 2)):
                        ng = pad_jn[ng_i : ng_i + 3]
                        if ng not in seen_ng3:
                            seen_ng3.add(ng)
                            self.jamo_ngram3_idx[ng].append(i)
                if j_norm_list:
                    self.synth_jamo_raw[i] = tuple(j_raw_list)
                    self.synth_jamo_norm[i] = tuple(j_norm_list)
                    self.synth_jamo_wtoks[i] = tuple(j_wtoks_list)

        valid_mask = self.gt > 0
        self.sorted_gt_indices = np.argsort(self.gt[valid_mask])
        valid_positions = np.flatnonzero(valid_mask)
        self.sorted_gt_cat_idx = valid_positions[self.sorted_gt_indices]
        self.sorted_gt_vals = self.gt[self.sorted_gt_cat_idx]
        self.sorted_gt_alphas: List[str] = [self.word_Alpha[idx] for idx in self.sorted_gt_cat_idx]
        self.sorted_gt_sorted_alphas: List[str] = [self.sorted_word_Alpha[idx] for idx in self.sorted_gt_cat_idx]

    def query_tonnage_window(self, ton: float, rel_tol: float = 0.06, max_cands: Optional[int] = None) -> List[int]:
        if ton <= 0 or len(self.sorted_gt_vals) == 0:
            return []
        low = ton * (1.0 - rel_tol) - 2.0
        high = ton * (1.0 + rel_tol) + 2.0
        left = int(np.searchsorted(self.sorted_gt_vals, low, side="left"))
        right = int(np.searchsorted(self.sorted_gt_vals, high, side="right"))
        if right <= left:
            return []
        if max_cands is None or (right - left) <= max_cands:
            return self.sorted_gt_cat_idx[left:right].tolist()
        center = int(np.searchsorted(self.sorted_gt_vals, ton))
        half = max_cands // 2
        s = max(left, center - half)
        e = min(right, s + max_cands)
        s = max(left, e - max_cands)
        return self.sorted_gt_cat_idx[s:e].tolist()

    def query_l_win_top_osa(
        self,
        ton: float,
        query_alpha: str,
        sorted_query_alpha: str,
        rel_tol: float = 0.10,
        top_k: int = 40,
    ) -> List[int]:
        if ton <= 0 or not query_alpha or len(self.sorted_gt_vals) == 0:
            return []
        low = ton * (1.0 - rel_tol) - 2.0
        high = ton * (1.0 + rel_tol) + 2.0
        left = int(np.searchsorted(self.sorted_gt_vals, low, side="left"))
        right = int(np.searchsorted(self.sorted_gt_vals, high, side="right"))
        if right <= left:
            return []
        if (right - left) <= top_k * 2:
            return self.sorted_gt_cat_idx[left:right].tolist()

        sub_alphas = self.sorted_gt_alphas[left:right]
        hits = process.extract(
            query_alpha,
            sub_alphas,
            scorer=OSA.normalized_similarity,
            limit=top_k,
            score_cutoff=0.52,
        )
        out = [int(self.sorted_gt_cat_idx[left + h[2]]) for h in hits]

        if sorted_query_alpha and sorted_query_alpha != query_alpha:
            sub_s_alphas = self.sorted_gt_sorted_alphas[left:right]
            s_hits = process.extract(
                sorted_query_alpha,
                sub_s_alphas,
                scorer=OSA.normalized_similarity,
                limit=25,
                score_cutoff=0.58,
            )
            for h in s_hits:
                out.append(int(self.sorted_gt_cat_idx[left + h[2]]))
        return out

    def query_jamo_top_candidates(self, q_jamo_norm: str, top_k: int = 250) -> List[int]:
        if not q_jamo_norm or not self.jamo_ngram3_idx:
            return []
        scores: Dict[int, float] = defaultdict(float)
        pad_q = f" {q_jamo_norm} "
        seen_ng = set()
        for i in range(max(0, len(pad_q) - 2)):
            ng = pad_q[i : i + 3]
            if ng in seen_ng:
                continue
            seen_ng.add(ng)
            posting = self.jamo_ngram3_idx.get(ng)
            if not posting or len(posting) > 2500:
                continue
            idf = 1.5 / math.sqrt(len(posting))
            for cand in posting:
                scores[cand] += idf
        q_len = len(q_jamo_norm)
        for cand in self.jamo_first_char_idx.get(q_jamo_norm[0], ()):
            c_norms = self.synth_jamo_norm[cand]
            if c_norms and abs(len(c_norms[0]) - q_len) <= 6:
                scores[cand] += 0.20
        if not scores:
            return []
        return [idx for idx, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def predict_rows(
    df: pd.DataFrame,
    model: Dict[str, Any],
    cat_idx: CatalogIndex,
    use_train_lookup: Optional[bool] = None,
    gold_ids: Optional[List[str]] = None,
    top_candidates: Optional[int] = None,
    jamo_score_mode: Optional[str] = None,
    candidate_diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    # Legacy is deliberately the default until controlled comparisons pass.
    jamo_score_mode = jamo_score_mode or str(model.get("jamo_score_mode", "legacy"))
    if jamo_score_mode not in ("legacy", "separate"):
        raise ValueError("jamo_score_mode must be legacy or separate")
    cands_list: List[List[Tuple[float, str]]] = []

    def capture_candidates(candidates, row_index):
        if candidate_diagnostics is not None:
            encoded = np.asarray(sorted(candidates), dtype="<i8").tobytes()
            candidate_diagnostics.append({
                "row_index": row_index, "count": len(candidates),
                "index_set_sha256": hashlib.sha256(encoded).hexdigest(),
            })
    if use_train_lookup is None:
        use_train_lookup = bool(model.get("use_train_lookup_default", USE_TRAIN_LOOKUP))

    kr_mode = str(model.get("kr_mode", KR_MODE))
    en_mode = str(model.get("en_mode", EN_MODE))

    callsign_lookup = model.get("unique_callsign_lookup") or model.get("callsign_lookup", {})
    name_lookup = model.get("unique_name_lookup") or model.get("name_lookup", {})
    word_map = model.get("learned_word_map", {})
    word_top2 = model.get("learned_word_top2", {})
    syl_top2 = model.get("learned_syl_top2", {})
    syl_top3 = model.get("learned_syl_top3", syl_top2)
    type_lift = model.get("type_log_lift", {})
    flag_lift = model.get("flag_log_lift", {})
    type_trans_prob = model.get("type_trans_prob", {})
    flag_trans_prob = model.get("flag_trans_prob", {})

    preds: List[str] = []
    gold_ranks: List[int] = []

    names = df["vessel_name"].tolist()
    callsigns = df["call_sign"].tolist() if "call_sign" in df.columns else [None] * len(df)
    gts = pd.to_numeric(df["gross_tonnage"], errors="coerce").fillna(-1.0).tolist()
    igts = pd.to_numeric(df["international_gross_tonnage"], errors="coerce").fillna(-1.0).tolist()
    flags = df["flag_name"].tolist() if "flag_name" in df.columns else [None] * len(df)
    vtypes = df["vessel_type"].tolist() if "vessel_type" in df.columns else [None] * len(df)
    years = df["arrival_year"].tolist() if "arrival_year" in df.columns else [None] * len(df)

    for row_i in range(len(df)):
        raw_name = names[row_i]
        cs = str(callsigns[row_i]).strip().upper() if pd.notna(callsigns[row_i]) else ""
        gt = gts[row_i]
        igt = igts[row_i]
        ton = igt if igt > 0 else (gt if gt > 0 else -1.0)
        alt_ton = gt if (gt > 0 and gt != ton) else -1.0

        kf = str(flags[row_i]).strip() if pd.notna(flags[row_i]) else "UNK"
        kt = str(vtypes[row_i]).strip() if pd.notna(vtypes[row_i]) else "UNK"
        yb = year_bucket(years[row_i])
        raw_yr = years[row_i]
        try:
            arrival_yr = float(raw_yr) if pd.notna(raw_yr) else None
        except (ValueError, TypeError):
            arrival_yr = None
        gold_vid = gold_ids[row_i] if gold_ids is not None else None

        kr_flag = is_korean_name(raw_name)

        def _is_tonnage_compatible(cand_vid: str, rel_tol: float) -> bool:
            c_idx = cat_idx.vid_to_idx.get(cand_vid)
            if c_idx is None:
                return True
            c_gt = cat_idx.gt[c_idx]
            if ton <= 0 or c_gt <= 0:
                return True
            diff = abs(ton - c_gt)
            if alt_ton > 0:
                diff = min(diff, abs(alt_ton - c_gt))
            return (diff / max(1.0, c_gt)) <= rel_tol

        if use_train_lookup:
            if cs and cs in callsign_lookup and _is_tonnage_compatible(callsign_lookup[cs], 0.12):
                pred_vid = callsign_lookup[cs]
                preds.append(pred_vid)
                if top_candidates is not None:
                    cands_list.append([(100.0, pred_vid)])
                gold_ranks.append(1 if (gold_vid is None or pred_vid == gold_vid) else 0)
                continue

        if not kr_flag:
            comp, wtoks, ntoks = normalize_vessel_name(raw_name, kr_mode="AR02")
            if use_train_lookup and comp and comp in name_lookup and _is_tonnage_compatible(name_lookup[comp], 0.06):
                pred_vid = name_lookup[comp]
                preds.append(pred_vid)
                if top_candidates is not None:
                    cands_list.append([(100.0, pred_vid)])
                gold_ranks.append(1 if (gold_vid is None or pred_vid == gold_vid) else 0)
                continue

            nstr = "|".join(ntoks)
            en_wtoks1 = wtoks
            sorted_en_wtoks = sorted(wtoks)
            query_alpha1 = "".join(wtoks)
            sorted_query_alpha = "".join(sorted_en_wtoks)
            query_skel1 = consonant_skeleton(query_alpha1)
            query_phon1 = "".join(en_to_phonetic(w) for w in wtoks)

            q_ord_nums = list(ntoks)
            if wtoks and wtoks[-1] in EN_NUM_WORD_MAP:
                q_ord_nums.append(EN_NUM_WORD_MAP[wtoks[-1]])
            nstr_ord = "|".join(q_ord_nums)

            cand_set = set()
            if comp in cat_idx.exact_compact_idx:
                cand_set.update(cat_idx.exact_compact_idx[comp])
            if query_alpha1 in cat_idx.exact_alpha_idx:
                cand_set.update(cat_idx.exact_alpha_idx[query_alpha1])
            if en_mode == "EN_V3" and sorted_query_alpha in cat_idx.exact_sorted_alpha_idx:
                cand_set.update(cat_idx.exact_sorted_alpha_idx[sorted_query_alpha])

            for w in en_wtoks1:
                if len(w) >= 5:
                    cand_set.update(cat_idx.outer2_idx.get(f"{w[:2]}_{w[-2:]}", []))
                if len(w) >= 3:
                    cand_set.update(cat_idx.word_idx.get(w[:4], []))
            if query_alpha1:
                cand_set.update(cat_idx.prefix3_idx.get(query_alpha1[:3], []))
                if en_mode == "EN_V3" and len(query_alpha1) >= 4:
                    cand_set.update(cat_idx.suffix3_idx.get(query_alpha1[-3:], []))
                if len(cand_set) < 200:
                    cand_set.update(cat_idx.prefix2_idx.get(query_alpha1[:2], []))
            if query_skel1:
                cand_set.update(cat_idx.skel3_idx.get(query_skel1[:3], []))
            if query_phon1:
                cand_set.update(cat_idx.phon3_idx.get(query_phon1[:3], []))

            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.10, max_cands=500))
                if en_mode == "EN_V3":
                    cand_set.update(cat_idx.query_l_win_top_osa(ton, query_alpha1, sorted_query_alpha, rel_tol=0.10, top_k=40))
                if alt_ton > 0:
                    cand_set.update(cat_idx.query_tonnage_window(alt_ton, rel_tol=0.10, max_cands=400))
                if len(cand_set) < 200:
                    cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.20, max_cands=500))

            if nstr and nstr in cat_idx.num_idx:
                num_cands = cat_idx.num_idx[nstr]
                if len(num_cands) <= 500:
                    cand_set.update(num_cands)
            if not cand_set:
                if ton > 0:
                    cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.35, max_cands=300))
                if not cand_set:
                    cand_set.add(0)

            t_lifts = type_lift.get(kt, {})
            f_lifts = flag_lift.get(f"{kf}|{yb}", flag_lift.get(f"{kf}|ALL", {}))
            n_q_words = len(en_wtoks1)

            capture_candidates(cand_set, row_i)
            scored_cands: List[Tuple[float, int]] = []
            for c_i in cand_set:
                c_alpha = cat_idx.word_Alpha[c_i]
                c_s_alpha = cat_idx.sorted_word_Alpha[c_i]
                c_skel = cat_idx.skeletons[c_i]
                c_nstr = cat_idx.num_strs[c_i]
                c_nstr_ord = cat_idx.num_strs_ord[c_i]
                c_gt = cat_idx.gt[c_i]
                c_wtoks = cat_idx.word_tokens[c_i]
                c_s_wtoks = cat_idx.sorted_word_tokens[c_i]

                if query_alpha1 and c_alpha:
                    sim_alpha = 1.0 if query_alpha1 == c_alpha else OSA.normalized_similarity(query_alpha1, c_alpha)
                else:
                    sim_alpha = 0.0

                if en_mode == "EN_V3" and n_q_words >= 2 and len(c_wtoks) >= 2 and sorted_query_alpha and c_s_alpha:
                    sim_s_alpha = 1.0 if sorted_query_alpha == c_s_alpha else OSA.normalized_similarity(sorted_query_alpha, c_s_alpha)
                    if 0.96 * sim_s_alpha > sim_alpha:
                        sim_alpha = 0.96 * sim_s_alpha

                sim_skel = OSA.normalized_similarity(query_skel1, c_skel) if (query_skel1 and c_skel) else 0.0

                typo_bonus = 0.0
                if n_q_words > 0 and n_q_words == len(c_wtoks):
                    w_sims = []
                    for a, b in zip(en_wtoks1, c_wtoks):
                        if a == b:
                            w_sims.append(1.0)
                        else:
                            t_kind = is_en_inner_typo(a, b)
                            if t_kind == 1:
                                w_sims.append(0.985)
                                typo_bonus += 0.45
                            elif t_kind == 2:
                                w_sims.append(0.95)
                                typo_bonus += 0.25
                            else:
                                w_sims.append(OSA.normalized_similarity(a, b))
                    sim_tok = sum(w_sims) / n_q_words

                    if en_mode == "EN_V3" and n_q_words >= 2 and sim_tok < 0.85:
                        sw_sims = []
                        for a, b in zip(sorted_en_wtoks, c_s_wtoks):
                            if a == b:
                                sw_sims.append(1.0)
                            elif is_en_inner_typo(a, b) == 1:
                                sw_sims.append(0.985)
                            else:
                                sw_sims.append(OSA.normalized_similarity(a, b))
                        sim_tok = max(sim_tok, 0.96 * (sum(sw_sims) / n_q_words))
                else:
                    sim_tok = sim_alpha

                name_score = 8.0 * sim_alpha + 1.0 * sim_skel + 2.0 * sim_tok + min(0.75, typo_bonus)
                if comp and comp == cat_idx.compact_names[c_i]:
                    name_score += 2.5

                num_score = 0.0
                if (nstr and c_nstr and nstr == c_nstr) or (nstr_ord and c_nstr_ord and nstr_ord == c_nstr_ord):
                    num_score = 3.2
                elif nstr and c_nstr and nstr != c_nstr:
                    num_score = -4.0
                elif nstr and not c_nstr_ord:
                    num_score = -2.2
                elif not nstr_ord and c_nstr:
                    num_score = -1.2

                ton_score = 0.0
                if ton > 0 and c_gt > 0:
                    diff = abs(ton - c_gt)
                    if alt_ton > 0:
                        diff = min(diff, abs(alt_ton - c_gt))
                    rel = diff / max(1.0, c_gt)
                    if diff <= 1.0:
                        ton_score = 3.6
                    elif rel <= 0.005:
                        ton_score = 3.1
                    elif rel <= 0.02:
                        ton_score = 2.5
                    elif rel <= 0.052:
                        ton_score = 2.1 - 4.0 * (rel - 0.02)
                    elif rel <= 0.095:
                        ton_score = 0.8 - 10.0 * (rel - 0.052)
                    elif rel <= 0.18:
                        ton_score = 0.2 - 4.0 * (rel - 0.095)
                    else:
                        ton_score = -2.2 * min(2.5, rel * 6.0)

                ct = cat_idx.types[c_i]
                cf = cat_idx.flags[c_i]
                default_t_pen = 0.0 if en_mode == "EN_V3" else -0.15
                t_val = t_lifts.get(ct, default_t_pen)
                if kf == "UNK":
                    f_val = 0.0
                else:
                    default_f_pen = 0.0
                    f_val = f_lifts.get(cf, default_f_pen)
                    if yb == "OLD" and f_val < 0:
                        f_val *= 0.25
                meta_score = 0.40 * t_val + 0.32 * f_val

                scored_cands.append((name_score + num_score + ton_score + meta_score, c_i))

            scored_cands.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))
            best_vid = cat_idx.vessel_ids[scored_cands[0][1]]
            preds.append(best_vid)
            if top_candidates is not None:
                cands_list.append([
                    (round(float(sc), 6), cat_idx.vessel_ids[c_i])
                    for sc, c_i in scored_cands[:top_candidates]
                ])
            if gold_vid is not None:
                grank = 0
                for r_idx, (_, c_i) in enumerate(scored_cands, start=1):
                    if cat_idx.vessel_ids[c_i] == gold_vid:
                        grank = r_idx
                        break
                gold_ranks.append(grank)
            else:
                gold_ranks.append(0)
            continue

        comp, wtoks, ntoks, kr_ord_nstr, cond_a_2char_ho = parse_kr_row_features(raw_name, kr_mode=kr_mode)
        if use_train_lookup and comp and comp in name_lookup and _is_tonnage_compatible(name_lookup[comp], 0.06):
            pred_vid = name_lookup[comp]
            preds.append(pred_vid)
            if top_candidates is not None:
                cands_list.append([(100.0, pred_vid)])
            gold_ranks.append(1 if (gold_vid is None or pred_vid == gold_vid) else 0)
            continue

        nstr = "|".join(ntoks)

        cands_pairs = [kr_word_to_en_candidates(w, syl_top2, word_map) for w in wtoks]
        en_wtoks1 = [p[0] for p in cands_pairs]
        en_wtoks2 = [p[1] for p in cands_pairs]
        query_alpha1 = "".join(en_wtoks1)
        query_alpha2 = "".join(en_wtoks2)
        query_phon_words = [
            en_to_phonetic(w_en) if (w in word_map or not re_hangul.search(w_en)) else kr_to_phonetic(w)
            for w, w_en in zip(wtoks, en_wtoks1)
        ]
        query_phon_full = "".join(query_phon_words)
        query_skel1 = consonant_skeleton(query_alpha1)
        query_skel2 = consonant_skeleton(query_alpha2)

        expanded_wtoks: List[str] = []
        for w in wtoks:
            acr = extract_kr_acronym_prefix(w)
            if acr is not None and acr[1]:
                expanded_wtoks.append(acr[0])
                expanded_wtoks.append(acr[1])
            elif acr is not None and not acr[1]:
                expanded_wtoks.append(w)
            else:
                expanded_wtoks.append(w)

        idea_b_word_cands = [generate_idea_b_word_variants(w, syl_top3, word_top2) for w in expanded_wtoks]
        idea_b_full_variants: List[str] = []
        for v_idx in range(8):
            full_v = "".join(wc[min(v_idx, len(wc) - 1)] for wc in idea_b_word_cands if wc)
            if full_v and full_v not in idea_b_full_variants:
                idea_b_full_variants.append(full_v)

        full_kr_concat = "".join(wtoks)
        if full_kr_concat in IDEA_B_WORD_VARS:
            for fv_direct in IDEA_B_WORD_VARS[full_kr_concat]:
                if fv_direct not in idea_b_full_variants:
                    idea_b_full_variants.insert(0, fv_direct)

        wtoks_no_ord_cands: Optional[List[List[str]]] = None
        if kr_ord_nstr and len(expanded_wtoks) >= 2:
            wtoks_no_ord_cands = idea_b_word_cands[:-1]
            for v_idx in range(3):
                full_v_no_ord = "".join(wc[min(v_idx, len(wc) - 1)] for wc in wtoks_no_ord_cands if wc)
                if full_v_no_ord and full_v_no_ord not in idea_b_full_variants:
                    idea_b_full_variants.append(full_v_no_ord)

        raw_clean = str(raw_name).strip()
        ends_with_ho_raw = raw_clean.endswith("호") and bool(wtoks) and not ntoks
        ho_preserved_alpha: Optional[str] = (query_alpha1 + "HO") if ends_with_ho_raw else None
        ho_preserved_phon: Optional[str] = (query_phon_full + en_to_phonetic("HO")) if ends_with_ho_raw else None
        if cond_a_2char_ho and ho_preserved_alpha and ho_preserved_alpha not in idea_b_full_variants:
            idea_b_full_variants.append(ho_preserved_alpha)

        q_jamo_wtoks = [hangul_to_jamo(w, phonetic_norm=True) for w in wtoks if w]
        q_jamo_raw = "".join(hangul_to_jamo(w, phonetic_norm=False) for w in wtoks if w)
        q_jamo_norm = "".join(q_jamo_wtoks)

        cand_set = set()
        if q_jamo_norm and cat_idx.jamo_ngram3_idx:
            if q_jamo_norm in cat_idx.exact_jamo_idx:
                cand_set.update(cat_idx.exact_jamo_idx[q_jamo_norm])
            if len(q_jamo_norm) >= 3:
                cand_set.update(cat_idx.jamo_prefix3_idx.get(q_jamo_norm[:3], []))
            if len(q_jamo_norm) >= 4:
                cand_set.update(cat_idx.jamo_suffix3_idx.get(q_jamo_norm[-3:], []))
            for jw in q_jamo_wtoks:
                if len(jw) >= 3:
                    cand_set.update(cat_idx.jamo_word_idx.get(jw[:4], []))
            cand_set.update(cat_idx.query_jamo_top_candidates(q_jamo_norm, top_k=250))

        if query_alpha1 in cat_idx.exact_alpha_idx:
            cand_set.update(cat_idx.exact_alpha_idx[query_alpha1])
        if ton > 0:
            cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.098, max_cands=None))
            cand_set.update(cat_idx.query_l_win_top_osa(ton, query_alpha1, "", rel_tol=0.10, top_k=40))
        if alt_ton > 0:
            cand_set.update(cat_idx.query_tonnage_window(alt_ton, rel_tol=0.098, max_cands=None))
        if query_alpha1:
            cand_set.update(cat_idx.prefix3_idx.get(query_alpha1[:3], []))
        if query_skel1:
            cand_set.update(cat_idx.skel3_idx.get(query_skel1[:3], []))
        if query_phon_full:
            cand_set.update(cat_idx.phon3_idx.get(query_phon_full[:3], []))
        for w in en_wtoks1:
            if len(w) >= 3:
                cand_set.update(cat_idx.word_idx.get(w[:4], []))
        for fv in idea_b_full_variants[:7]:
            if fv in cat_idx.exact_alpha_idx:
                cand_set.update(cat_idx.exact_alpha_idx[fv])
            if len(fv) >= 3:
                cand_set.update(cat_idx.prefix3_idx.get(fv[:3], []))
        for wc in idea_b_word_cands:
            for vw in wc[:5]:
                if len(vw) >= 3:
                    cand_set.update(cat_idx.word_idx.get(vw[:4], []))
        if nstr and nstr in cat_idx.num_idx:
            num_cands = cat_idx.num_idx[nstr]
            if len(num_cands) <= 500:
                cand_set.update(num_cands)
        if kr_ord_nstr and kr_ord_nstr in cat_idx.num_idx:
            num_cands_ord = cat_idx.num_idx[kr_ord_nstr]
            if len(num_cands_ord) <= 500:
                cand_set.update(num_cands_ord)
        if not cand_set:
            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.30, max_cands=200))
            if not cand_set:
                cand_set.add(0)

        t_lifts = type_lift.get(kt, {})
        f_lifts = flag_lift.get(f"{kf}|{yb}", flag_lift.get(f"{kf}|ALL", {}))
        n_q_words = len(en_wtoks1)
        n_b_words = len(idea_b_word_cands)

        capture_candidates(cand_set, row_i)
        scored_cands = []
        best_jamo_by_ci: Dict[int, float] = {}
        for c_i in cand_set:
            c_alpha = cat_idx.word_Alpha[c_i]
            c_skel = cat_idx.skeletons[c_i]
            c_nstr = cat_idx.num_strs[c_i]
            c_nstr_ord = cat_idx.num_strs_ord[c_i]
            c_gt = cat_idx.gt[c_i]
            c_wtoks = cat_idx.word_tokens[c_i]
            c_pwords = cat_idx.phon_words[c_i]
            c_pfull = cat_idx.phon_full[c_i]

            sim_a1 = OSA.normalized_similarity(query_alpha1, c_alpha) if (query_alpha1 and c_alpha) else 0.0
            sim_a2 = OSA.normalized_similarity(query_alpha2, c_alpha) if (query_alpha2 and c_alpha) else 0.0
            sim_alpha = max(sim_a1, 0.97 * sim_a2)

            sim_s1 = OSA.normalized_similarity(query_skel1, c_skel) if (query_skel1 and c_skel) else 0.0
            sim_s2 = OSA.normalized_similarity(query_skel2, c_skel) if (query_skel2 and c_skel) else 0.0
            sim_skel = max(sim_s1, sim_s2)

            sim_phon = OSA.normalized_similarity(query_phon_full, c_pfull) if (query_phon_full and c_pfull) else 0.0

            allow_ho_match = ends_with_ho_raw and (cond_a_2char_ho or (len(c_wtoks) > 0 and c_wtoks[-1] == "HO"))
            if allow_ho_match and ho_preserved_alpha and c_alpha:
                sim_alpha = max(sim_alpha, OSA.normalized_similarity(ho_preserved_alpha, c_alpha))
                if ho_preserved_phon and c_pfull:
                    sim_phon = max(sim_phon, OSA.normalized_similarity(ho_preserved_phon, c_pfull))

            if idea_b_full_variants and c_alpha:
                best_b_full = max(OSA.normalized_similarity(fv, c_alpha) for fv in idea_b_full_variants)
                sim_alpha = max(sim_alpha, best_b_full)
                best_b_skel = max(
                    OSA.normalized_similarity(consonant_skeleton(fv), c_skel)
                    for fv in idea_b_full_variants[:5]
                ) if c_skel else 0.0
                sim_skel = max(sim_skel, best_b_skel)
                best_b_phon = max(
                    OSA.normalized_similarity(en_to_phonetic(fv), c_pfull)
                    for fv in idea_b_full_variants[:4]
                ) if c_pfull else 0.0
                sim_phon = max(sim_phon, 0.96 * best_b_phon)

            matched_word_count = False
            sim_tok = max(sim_alpha, sim_phon)

            if n_q_words > 0 and n_q_words == len(c_wtoks):
                w_sims = []
                for qw1, qw2, qpw, cw, cpw in zip(en_wtoks1, en_wtoks2, query_phon_words, c_wtoks, c_pwords):
                    s_w = max(
                        OSA.normalized_similarity(qw1, cw),
                        0.97 * OSA.normalized_similarity(qw2, cw),
                        0.95 * OSA.normalized_similarity(qpw, cpw),
                    )
                    w_sims.append(s_w)
                sim_tok = sum(w_sims) / n_q_words
                matched_word_count = True

            if n_b_words > 0 and n_b_words == len(c_wtoks):
                bw_sims = []
                for wc_list, cw, cpw in zip(idea_b_word_cands, c_wtoks, c_pwords):
                    best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                    best_pw = max(
                        OSA.normalized_similarity(en_to_phonetic(cand_w), cpw)
                        for cand_w in wc_list[:4]
                    ) if cpw else 0.0
                    bw_sims.append(max(best_w, 0.95 * best_pw))
                sim_tok = max(sim_tok, sum(bw_sims) / n_b_words) if matched_word_count else (sum(bw_sims) / n_b_words)
                matched_word_count = True

            if not matched_word_count and wtoks_no_ord_cands is not None and len(wtoks_no_ord_cands) == len(c_wtoks) and len(c_wtoks) > 0 and c_nstr:
                bw_sims_no_ord = []
                for wc_list, cw, cpw in zip(wtoks_no_ord_cands, c_wtoks, c_pwords):
                    best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                    best_pw = max(
                        OSA.normalized_similarity(en_to_phonetic(cand_w), cpw)
                        for cand_w in wc_list[:3]
                    ) if cpw else 0.0
                    bw_sims_no_ord.append(max(best_w, 0.95 * best_pw))
                sim_tok = max(sim_tok, sum(bw_sims_no_ord) / len(c_wtoks))
                matched_word_count = True

            if not matched_word_count and allow_ho_match and (n_b_words + 1) == len(c_wtoks):
                cands_plus_ho = idea_b_word_cands + [["HO"]]
                bw_sims_ho = []
                for wc_list, cw, cpw in zip(cands_plus_ho, c_wtoks, c_pwords):
                    best_w = max(OSA.normalized_similarity(cand_w, cw) for cand_w in wc_list)
                    bw_sims_ho.append(best_w)
                sim_tok = max(sim_tok, sum(bw_sims_ho) / len(c_wtoks))
                matched_word_count = True

            jamo_first_ok = False
            jamo_bonus = 0.0
            if q_jamo_norm and cat_idx.synth_jamo_norm[c_i]:
                c_jraws = cat_idx.synth_jamo_raw[c_i]
                c_jnorms = cat_idx.synth_jamo_norm[c_i]
                c_jwtoks_list = cat_idx.synth_jamo_wtoks[c_i]
                sim_j_raw = max(
                    (1.0 if q_jamo_raw == jr else OSA.normalized_similarity(q_jamo_raw, jr))
                    for jr in c_jraws if jr
                ) if (q_jamo_raw and c_jraws) else 0.0
                sim_j_norm = max(
                    (1.0 if q_jamo_norm == jn else OSA.normalized_similarity(q_jamo_norm, jn))
                    for jn in c_jnorms if jn
                )
                sim_jamo = 0.60 * sim_j_raw + 0.40 * sim_j_norm
                if jamo_score_mode == "separate":
                    # One bounded channel; leave original romanized, metadata,
                    # tonnage-gating and word-count features untouched.
                    jamo_bonus = 1.8 * sim_jamo
                else:
                    best_jamo_by_ci[c_i] = max(sim_jamo, sim_j_norm)
                    sim_alpha = max(sim_alpha, sim_jamo)
                    sim_phon = max(sim_phon, sim_j_norm)
                    sim_skel = max(sim_skel, sim_j_norm)
                    jamo_first_ok = any(jn and q_jamo_norm[0] == jn[0] for jn in c_jnorms)
                    for cj_wtoks in c_jwtoks_list:
                        if len(q_jamo_wtoks) > 0 and len(q_jamo_wtoks) == len(cj_wtoks):
                            j_w_sim = sum(
                                1.0 if a == b else OSA.normalized_similarity(a, b)
                                for a, b in zip(q_jamo_wtoks, cj_wtoks)
                            ) / len(q_jamo_wtoks)
                            sim_tok = max(sim_tok, j_w_sim) if matched_word_count else max(sim_jamo, j_w_sim)
                            matched_word_count = True
                        elif sim_jamo > sim_tok:
                            sim_tok = sim_jamo

            if matched_word_count:
                word_cnt_bonus = 0.45
            else:
                if sim_alpha >= 0.90:
                    word_cnt_bonus = 0.25
                elif sim_alpha >= 0.82 or sim_phon >= 0.85:
                    word_cnt_bonus = 0.0
                else:
                    word_cnt_bonus = -0.35 * min(2, abs(n_q_words - len(c_wtoks)))

            first_match = 0.35 if (jamo_first_ok or (query_phon_full and c_pfull and query_phon_full[0] == c_pfull[0])) else 0.0

            name_score = (
                3.8 * sim_alpha
                + 2.8 * sim_phon
                + 2.2 * sim_skel
                + 2.7 * sim_tok
                + word_cnt_bonus
                + first_match
            )

            num_score = 0.0
            if nstr:
                if c_nstr_ord:
                    num_score = 3.2 if (nstr == c_nstr or nstr == c_nstr_ord) else -4.0
                else:
                    num_score = -2.8
            elif kr_ord_nstr:
                if c_nstr or c_nstr_ord:
                    num_score = 3.2 if (kr_ord_nstr == c_nstr or kr_ord_nstr == c_nstr_ord) else -4.0
                else:
                    num_score = 0.0
            else:
                if c_nstr:
                    num_score = -1.6

            ton_score = 0.0
            if ton > 0 and c_gt > 0:
                diff = abs(ton - c_gt)
                if alt_ton > 0:
                    diff = min(diff, abs(alt_ton - c_gt))
                rel = diff / max(1.0, c_gt)
                if diff <= 1.0:
                    ton_score = 3.6
                elif rel <= 0.005:
                    ton_score = 3.1
                elif rel <= 0.02:
                    ton_score = 2.5
                elif rel <= 0.052:
                    ton_score = 2.1 - 4.0 * (rel - 0.02)
                elif rel <= 0.095:
                    ton_score = 0.8 - 10.0 * (rel - 0.052)
                elif kr_mode == "AR06" and (sim_alpha >= 0.88 or sim_phon >= 0.90) and rel <= 0.25:
                    ton_score = -0.8
                else:
                    ton_score = -3.5 * min(2.5, rel * 6.0)

            ct = cat_idx.types[c_i]
            cf = cat_idx.flags[c_i]

            if type_trans_prob and flag_trans_prob:
                if kt == "UNK" or kt not in type_trans_prob:
                    t_score = 0.5
                else:
                    p_t = type_trans_prob[kt].get(ct, 0.0)
                    if p_t >= 0.20:
                        t_score = 1.0
                    elif p_t >= 0.02:
                        t_score = p_t
                    else:
                        t_score = 0.0

                if kf == "UNK":
                    f_score = 0.5
                else:
                    fp_dict = flag_trans_prob.get(f"{kf}|{yb}", flag_trans_prob.get(f"{kf}|ALL", flag_trans_prob.get(kf, {})))
                    p_f = fp_dict.get(cf, 0.0) if fp_dict else 0.0
                    sim_for_flag = max(sim_alpha, sim_phon)
                    is_korea = cf in ("South Korea", "Korea", "Republic of Korea")
                    if is_korea or p_f >= 0.30:
                        f_score = 1.0
                    elif p_f >= 0.02:
                        f_score = p_f
                    elif sim_for_flag >= 0.85:
                        f_score = 0.5
                    elif sim_for_flag >= 0.70 and (arrival_yr is not None and arrival_yr <= 2019):
                        f_score = 0.5
                    else:
                        f_score = 0.0

                meta_score = 1.6 * (t_score - 0.5) + 1.2 * (f_score - 0.5)
            else:
                t_val = t_lifts.get(ct, -0.15)
                if kf == "UNK":
                    f_val = 0.0
                else:
                    f_val = f_lifts.get(cf, 0.0)
                    if yb == "OLD" and f_val < 0:
                        f_val *= 0.25
                meta_score = 0.40 * t_val + 0.32 * f_val

            total_score = name_score + num_score + ton_score + meta_score + jamo_bonus
            scored_cands.append((total_score, c_i))

        scored_cands.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))

        if len(scored_cands) > 1 and re_hangul.search(full_kr_concat):
            top_k = min(15, len(scored_cands))
            refined: List[Tuple[float, int]] = []
            for base_sc, c_i in scored_cands[:top_k]:
                c_alpha = cat_idx.word_Alpha[c_i]
                c_wtoks = cat_idx.word_tokens[c_i]
                lat_sim = max(
                    lattice_word_similarity(full_kr_concat, c_alpha, syl_top3),
                    best_jamo_by_ci.get(c_i, 0.0),
                )
                if kr_ord_nstr and len(wtoks) >= 2 and c_idx_has_num(cat_idx, c_i):
                    lat_sim = max(lat_sim, lattice_word_similarity("".join(wtoks[:-1]), c_alpha, syl_top3))
                allow_ho_lat = ends_with_ho_raw and (cond_a_2char_ho or (len(c_wtoks) > 0 and c_wtoks[-1] == "HO"))
                if allow_ho_lat and c_alpha.endswith("HO"):
                    lat_sim = max(lat_sim, lattice_word_similarity(full_kr_concat + "호", c_alpha, syl_top3))
                refined.append((base_sc + 1.8 * lat_sim, c_i))
            refined.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))
            scored_cands = refined + scored_cands[top_k:]

        best_vid = cat_idx.vessel_ids[scored_cands[0][1]]
        preds.append(best_vid)

        if top_candidates is not None:
            cands_list.append([
                (round(float(sc), 6), cat_idx.vessel_ids[c_i])
                for sc, c_i in scored_cands[:top_candidates]
            ])

        if gold_vid is not None:
            grank = 0
            for rank_idx, (_, c_i) in enumerate(scored_cands, start=1):
                if cat_idx.vessel_ids[c_i] == gold_vid:
                    grank = rank_idx
                    break
            gold_ranks.append(grank)
        else:
            gold_ranks.append(0)

    if top_candidates is not None:
        return preds, gold_ranks, cands_list
    return preds, gold_ranks


def c_idx_has_num(cat_idx: CatalogIndex, c_i: int) -> bool:
    return bool(cat_idx.num_strs[c_i] or cat_idx.num_strs_ord[c_i])


def resolve_input_dir(base_dir: Path) -> Path:
    for cand in [base_dir / "data", base_dir / "open", base_dir.parent / "data", base_dir.parent / "open"]:
        if (cand / "test.csv").exists() and (cand / "vessel_catalog.csv").exists():
            return cand
    raise FileNotFoundError("test.csv 및 vessel_catalog.csv를 data/ 또는 open/에서 찾을 수 없습니다.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_input_dir(base_dir)
    model_path = base_dir / "model" / "model.json"
    out_dir = base_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    catalog_df = pd.read_csv(data_dir / "vessel_catalog.csv", encoding="utf-8-sig")

    cat_idx = CatalogIndex(catalog_df, full_ko_map=model.get("catalog_full_ko_map"))
    use_lookup = bool(model.get("use_train_lookup_default", USE_TRAIN_LOOKUP))

    preds, _ = predict_rows(test_df, model, cat_idx, use_train_lookup=use_lookup, gold_ids=None)

    sub = pd.DataFrame({
        "sample_id": test_df["sample_id"].astype(str),
        "target_vessel_id": preds,
    })
    sub.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()