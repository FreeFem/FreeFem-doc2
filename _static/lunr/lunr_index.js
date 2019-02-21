const LUNR_DATA = [{"version":"2.3.5","fields":["t","b"],"fieldVectors":[["t/0",[0,2.697,1,2.697,2,3.436]],["b/0",[]],["t/1",[3,4.417]],["b/1",[]],["t/2",[4,4.098,5,3.216]],["b/2",[]],["t/3",[6,4.417]],["b/3",[]],["t/4",[7,5.075]],["b/4",[]],["t/5",[8,2.991,9,3.436,10,3.436]],["b/5",[]],["t/6",[11,5.075]],["b/6",[]],["t/7",[12,5.075]],["b/7",[]],["t/8",[13,3.566,14,3.566]],["b/8",[]],["t/9",[5,3.983]],["b/9",[]],["t/10",[15,3.566,16,3.566]],["b/10",[]],["t/11",[17,5.075]],["b/11",[]],["t/12",[18,4.417]],["b/12",[]],["t/13",[19,5.075]],["b/13",[]],["t/14",[20,4.417]],["b/14",[]],["t/15",[0,2.697,1,2.697,21,3.436]],["b/15",[]],["t/16",[3,4.417]],["b/16",[]],["t/17",[13,3.566,14,3.566]],["b/17",[]],["t/18",[6,4.417]],["b/18",[]],["t/19",[15,3.566,16,3.566]],["b/19",[]],["t/20",[22,5.075]],["b/20",[]],["t/21",[18,4.417]],["b/21",[]],["t/22",[20,4.417]],["b/22",[]],["t/23",[5,2.697,23,3.436,24,3.436]],["b/23",[]],["t/24",[25,5.075]],["b/24",[]],["t/25",[26,5.075]],["b/25",[]],["t/26",[27,5.075]],["b/26",[]],["t/27",[28,4.098,29,4.098]],["b/27",[]],["t/28",[30,5.075]],["b/28",[]],["t/29",[31,4.098,32,4.098]],["b/29",[]],["t/30",[33,2.959,34,2.959,35,2.959,36,2.959]],["b/30",[]],["t/31",[8,3.566,37,4.098]],["b/31",[]],["t/32",[38,3.436,39,3.436,40,2.303]],["b/32",[]],["t/33",[41,3.983]],["b/33",[]],["t/34",[40,2.746,42,4.098]],["b/34",[]],["t/35",[43,2.575,44,2.959,45,2.959,46,2.322]],["b/35",[]],["t/36",[40,2.303,47,3.436,48,3.436]],["b/36",[]],["t/37",[49,4.098,50,4.098]],["b/37",[]],["t/38",[51,2.697,52,2.303,53,2.303]],["b/38",[]],["t/39",[41,2.697,54,2.991,55,2.991]],["b/39",[]],["t/40",[40,1.983,54,2.575,55,2.575,56,2.575]],["b/40",[]],["t/41",[40,2.746,56,3.566]],["b/41",[]],["t/42",[46,3.216,57,4.098]],["b/42",[]],["t/43",[58,4.098,59,4.098]],["b/43",[]],["t/44",[60,3.436,61,3.436,62,3.436]],["b/44",[]],["t/45",[63,5.075]],["b/45",[]],["t/46",[64,4.098,65,4.098]],["b/46",[]],["t/47",[66,5.075]],["b/47",[]],["t/48",[67,4.098,68,4.098]],["b/48",[]],["t/49",[69,4.098,70,3.566]],["b/49",[]],["t/50",[71,5.075]],["b/50",[]],["t/51",[72,5.075]],["b/51",[]],["t/52",[73,4.098,74,4.098]],["b/52",[]],["t/53",[75,5.075]],["b/53",[]],["t/54",[70,4.417]],["b/54",[]],["t/55",[43,2.991,46,2.697,76,3.436]],["b/55",[]],["t/56",[77,5.075]],["b/56",[]],["t/57",[78,3.436,79,3.436,80,3.436]],["b/57",[]],["t/58",[41,3.216,81,3.566]],["b/58",[]],["t/59",[53,1.983,82,2.959,83,2.959,84,2.959]],["b/59",[]],["t/60",[85,2.315,86,2.315,87,2.315,88,2.015,89,2.015,90,2.315]],["b/60",[]],["t/61",[88,3.566,91,4.098]],["b/61",[]],["t/62",[53,1.551,92,2.315,93,2.315,94,2.315,95,2.015,96,2.015]],["b/62",[]],["t/63",[96,3.566,97,4.098]],["b/63",[]],["t/64",[98,5.075]],["b/64",[]],["t/65",[99,3.436,100,3.436,101,3.436]],["b/65",[]],["t/66",[102,5.075]],["b/66",[]],["t/67",[51,1.817,52,1.551,53,1.551,103,2.315,104,2.315,105,2.315]],["b/67",[]],["t/68",[0,2.039,51,2.039,52,1.741,53,1.741,106,2.597]],["b/68",[]],["t/69",[95,3.566,107,4.098]],["b/69",[]],["t/70",[108,3.436,109,3.436,110,3.436]],["b/70",[]],["t/71",[1,2.039,111,2.597,112,2.597,113,2.597,114,2.597]],["b/71",[]],["t/72",[52,2.303,81,2.991,115,3.436]],["b/72",[]],["t/73",[89,3.566,116,4.098]],["b/73",[]],["t/74",[52,1.399,117,2.088,118,2.088,119,2.088,120,2.088,121,2.088,122,2.088]],["b/74",[]],["t/75",[123,4.098,124,4.098]],["b/75",[]]],"invertedIndex":[["",{"_index":1,"t":{"0":{},"15":{},"71":{}},"b":{}}],["acousticssymbol",{"_index":77,"t":{"56":{}},"b":{}}],["algorithm",{"_index":0,"t":{"0":{},"15":{},"68":{}},"b":{}}],["authorssymbol",{"_index":25,"t":{"24":{}},"b":{}}],["blade",{"_index":87,"t":{"60":{}},"b":{}}],["boundari",{"_index":48,"t":{"36":{}},"b":{}}],["citationsymbol",{"_index":26,"t":{"25":{}},"b":{}}],["classif",{"_index":82,"t":{"59":{}},"b":{}}],["complex",{"_index":79,"t":{"57":{}},"b":{}}],["compress",{"_index":33,"t":{"30":{}},"b":{}}],["conductionsymbol",{"_index":116,"t":{"73":{}},"b":{}}],["contributingsymbol",{"_index":27,"t":{"26":{}},"b":{}}],["controlsymbol",{"_index":107,"t":{"69":{}},"b":{}}],["convect",{"_index":112,"t":{"71":{}},"b":{}}],["coupl",{"_index":45,"t":{"35":{}},"b":{}}],["dd)symbol",{"_index":10,"t":{"5":{}},"b":{}}],["decomposit",{"_index":9,"t":{"5":{}},"b":{}}],["decompositionsymbol",{"_index":37,"t":{"31":{}},"b":{}}],["depend",{"_index":93,"t":{"62":{}},"b":{}}],["developerssymbol",{"_index":3,"t":{"1":{},"16":{}},"b":{}}],["differenti",{"_index":84,"t":{"59":{}},"b":{}}],["documentationsymbol",{"_index":5,"t":{"2":{},"9":{},"23":{}},"b":{}}],["domain",{"_index":8,"t":{"5":{},"31":{}},"b":{}}],["download",{"_index":28,"t":{"27":{}},"b":{}}],["effectssymbol",{"_index":90,"t":{"60":{}},"b":{}}],["eigen",{"_index":38,"t":{"32":{}},"b":{}}],["elasticitysymbol",{"_index":41,"t":{"33":{},"39":{},"58":{}},"b":{}}],["elementsymbol",{"_index":14,"t":{"8":{},"17":{}},"b":{}}],["equationssymbol",{"_index":53,"t":{"38":{},"59":{},"62":{},"67":{},"68":{}},"b":{}}],["equationsymbol",{"_index":110,"t":{"70":{}},"b":{}}],["evolut",{"_index":42,"t":{"34":{}},"b":{}}],["exampl",{"_index":78,"t":{"57":{}},"b":{}}],["examplessymbol",{"_index":6,"t":{"3":{},"18":{}},"b":{}}],["exchangersymbol",{"_index":97,"t":{"63":{}},"b":{}}],["extern",{"_index":64,"t":{"46":{}},"b":{}}],["fan",{"_index":86,"t":{"60":{}},"b":{}}],["ffddm",{"_index":4,"t":{"2":{}},"b":{}}],["ffddmsymbol",{"_index":7,"t":{"4":{}},"b":{}}],["finit",{"_index":13,"t":{"8":{},"17":{}},"b":{}}],["flow",{"_index":88,"t":{"60":{},"61":{}},"b":{}}],["fluid",{"_index":43,"t":{"35":{},"55":{}},"b":{}}],["fluidssymbol",{"_index":115,"t":{"72":{}},"b":{}}],["formsymbol",{"_index":122,"t":{"74":{}},"b":{}}],["formulaesymbol",{"_index":74,"t":{"52":{}},"b":{}}],["free",{"_index":47,"t":{"36":{}},"b":{}}],["freefem",{"_index":24,"t":{"23":{}},"b":{}}],["freefem++symbol",{"_index":29,"t":{"27":{}},"b":{}}],["functionssymbol",{"_index":66,"t":{"47":{}},"b":{}}],["galleri",{"_index":61,"t":{"44":{}},"b":{}}],["generationsymbol",{"_index":16,"t":{"10":{},"19":{}},"b":{}}],["global",{"_index":67,"t":{"48":{}},"b":{}}],["guidesymbol",{"_index":32,"t":{"29":{}},"b":{}}],["heat",{"_index":96,"t":{"62":{},"63":{}},"b":{}}],["hillsymbol",{"_index":114,"t":{"71":{}},"b":{}}],["hookean",{"_index":35,"t":{"30":{}},"b":{}}],["i/osymbol",{"_index":63,"t":{"45":{}},"b":{}}],["inequalitysymbol",{"_index":59,"t":{"43":{}},"b":{}}],["instal",{"_index":31,"t":{"29":{}},"b":{}}],["introductionsymbol",{"_index":30,"t":{"28":{}},"b":{}}],["irrot",{"_index":85,"t":{"60":{}},"b":{}}],["languag",{"_index":69,"t":{"49":{}},"b":{}}],["larg",{"_index":76,"t":{"55":{}},"b":{}}],["librariessymbol",{"_index":65,"t":{"46":{}},"b":{}}],["linear",{"_index":55,"t":{"39":{},"40":{}},"b":{}}],["loopssymbol",{"_index":71,"t":{"50":{}},"b":{}}],["materialssymbol",{"_index":36,"t":{"30":{}},"b":{}}],["mathemat",{"_index":49,"t":{"37":{}},"b":{}}],["matlab",{"_index":100,"t":{"65":{}},"b":{}}],["matrix",{"_index":121,"t":{"74":{}},"b":{}}],["membranesymbol",{"_index":102,"t":{"66":{}},"b":{}}],["mesh",{"_index":15,"t":{"10":{},"19":{}},"b":{}}],["method",{"_index":104,"t":{"67":{}},"b":{}}],["miscsymbol",{"_index":22,"t":{"20":{}},"b":{}}],["modelssymbol",{"_index":50,"t":{"37":{}},"b":{}}],["modessymbol",{"_index":62,"t":{"44":{}},"b":{}}],["navier",{"_index":51,"t":{"38":{},"67":{},"68":{}},"b":{}}],["neo",{"_index":34,"t":{"30":{}},"b":{}}],["newton",{"_index":103,"t":{"67":{}},"b":{}}],["non",{"_index":54,"t":{"39":{},"40":{}},"b":{}}],["notationssymbol",{"_index":17,"t":{"11":{}},"b":{}}],["numberssymbol",{"_index":80,"t":{"57":{}},"b":{}}],["octavesymbol",{"_index":101,"t":{"65":{}},"b":{}}],["operatorssymbol",{"_index":72,"t":{"51":{}},"b":{}}],["optim",{"_index":95,"t":{"62":{},"69":{}},"b":{}}],["optimizationssymbol",{"_index":21,"t":{"15":{}},"b":{}}],["optimizationsymbol",{"_index":2,"t":{"0":{}},"b":{}}],["parallelizationsymbol",{"_index":18,"t":{"12":{},"21":{}},"b":{}}],["parameterssymbol",{"_index":11,"t":{"6":{}},"b":{}}],["partial",{"_index":83,"t":{"59":{}},"b":{}}],["plot",{"_index":99,"t":{"65":{}},"b":{}}],["pluginssymbol",{"_index":19,"t":{"13":{}},"b":{}}],["poisson’",{"_index":109,"t":{"70":{}},"b":{}}],["problemssymbol",{"_index":40,"t":{"32":{},"34":{},"36":{},"40":{},"41":{}},"b":{}}],["problemsymbol",{"_index":46,"t":{"35":{},"42":{},"55":{}},"b":{}}],["project",{"_index":106,"t":{"68":{}},"b":{}}],["propagationsymbol",{"_index":124,"t":{"75":{}},"b":{}}],["pure",{"_index":111,"t":{"71":{}},"b":{}}],["quadratur",{"_index":73,"t":{"52":{}},"b":{}}],["referencessymbol",{"_index":70,"t":{"49":{},"54":{}},"b":{}}],["rotat",{"_index":113,"t":{"71":{}},"b":{}}],["schema",{"_index":94,"t":{"62":{}},"b":{}}],["shockssymbol",{"_index":91,"t":{"61":{}},"b":{}}],["solv",{"_index":108,"t":{"70":{}},"b":{}}],["solver",{"_index":120,"t":{"74":{}},"b":{}}],["static",{"_index":56,"t":{"40":{},"41":{}},"b":{}}],["steadi",{"_index":105,"t":{"67":{}},"b":{}}],["stoke",{"_index":52,"t":{"38":{},"67":{},"68":{},"72":{},"74":{}},"b":{}}],["structur",{"_index":44,"t":{"35":{}},"b":{}}],["system",{"_index":81,"t":{"58":{},"72":{}},"b":{}}],["thermal",{"_index":89,"t":{"60":{},"73":{}},"b":{}}],["time",{"_index":92,"t":{"62":{}},"b":{}}],["transient",{"_index":119,"t":{"74":{}},"b":{}}],["transmiss",{"_index":57,"t":{"42":{}},"b":{}}],["tutori",{"_index":117,"t":{"74":{}},"b":{}}],["tutorialssymbol",{"_index":98,"t":{"64":{}},"b":{}}],["tutorialsymbol",{"_index":12,"t":{"7":{}},"b":{}}],["typessymbol",{"_index":75,"t":{"53":{}},"b":{}}],["valu",{"_index":39,"t":{"32":{}},"b":{}}],["variablessymbol",{"_index":68,"t":{"48":{}},"b":{}}],["variat",{"_index":58,"t":{"43":{}},"b":{}}],["visualizationsymbol",{"_index":20,"t":{"14":{},"22":{}},"b":{}}],["welcom",{"_index":23,"t":{"23":{}},"b":{}}],["whisper",{"_index":60,"t":{"44":{}},"b":{}}],["wifi",{"_index":123,"t":{"75":{}},"b":{}}],["write",{"_index":118,"t":{"74":{}},"b":{}}]],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]},
{"version":"2.3.5","fields":["t","b"],"fieldVectors":[],"invertedIndex":[],"pipeline":["stemmer"]}];
const PREVIEW_LOOKUP = [{"0":{"t":"Algorithms & OptimizationSymbole","p":"","l":"documentation/algorithmsOptimization.html"},"1":{"t":"DevelopersSymbole","p":"","l":"documentation/developers.html"},"2":{"t":"ffddm documentationSymbole","p":"","l":"documentation/ffddm/documentation.html"},"3":{"t":"ExamplesSymbole","p":"","l":"documentation/ffddm/examples.html"},"4":{"t":"ffddmSymbole","p":"","l":"documentation/ffddm/index.html"},"5":{"t":"Domain Decomposition (DD)Symbole","p":"","l":"documentation/ffddm/introddm.html"},"6":{"t":"ParametersSymbole","p":"","l":"documentation/ffddm/parameters.html"},"7":{"t":"TutorialSymbole","p":"","l":"documentation/ffddm/tutorial.html"},"8":{"t":"Finite elementSymbole","p":"","l":"documentation/finiteElement.html"},"9":{"t":"DocumentationSymbole","p":"","l":"documentation/index.html"},"10":{"t":"Mesh GenerationSymbole","p":"","l":"documentation/meshGeneration.html"},"11":{"t":"NotationsSymbole","p":"","l":"documentation/notations.html"},"12":{"t":"ParallelizationSymbole","p":"","l":"documentation/parallelization.html"},"13":{"t":"PluginsSymbole","p":"","l":"documentation/plugins.html"},"14":{"t":"VisualizationSymbole","p":"","l":"documentation/visualization.html"},"15":{"t":"Algorithms & OptimizationsSymbole","p":"","l":"example/algoOptimizations.html"},"16":{"t":"DevelopersSymbole","p":"","l":"example/developers.html"},"17":{"t":"Finite ElementSymbole","p":"","l":"example/finiteElement.html"},"18":{"t":"ExamplesSymbole","p":"","l":"example/index.html"},"19":{"t":"Mesh GenerationSymbole","p":"","l":"example/meshGeneration.html"},"20":{"t":"MiscSymbole","p":"","l":"example/misc.html"},"21":{"t":"ParallelizationSymbole","p":"","l":"example/parallelization.html"},"22":{"t":"VisualizationSymbole","p":"","l":"example/visualization.html"},"23":{"t":"Welcome to FreeFem++ documentationSymbole","p":"","l":"index.html"},"24":{"t":"AuthorsSymbole","p":"","l":"introduction/authors.html"},"25":{"t":"CitationSymbole","p":"","l":"introduction/citation.html"},"26":{"t":"ContributingSymbole","p":"","l":"introduction/contributing.html"},"27":{"t":"Download FreeFem++Symbole","p":"","l":"introduction/download.html"},"28":{"t":"IntroductionSymbole","p":"","l":"introduction/index.html"},"29":{"t":"Installation guideSymbole","p":"","l":"introduction/installation.html"},"30":{"t":"Compressible Neo-Hookean materialsSymbole","p":"","l":"model/compressibleNeoHookeanMaterials.html"},"31":{"t":"Domain decompositionSymbole","p":"","l":"model/domainDecomposition.html"},"32":{"t":"Eigen value problemsSymbole","p":"","l":"model/eigenValueProblems.html"},"33":{"t":"ElasticitySymbole","p":"","l":"model/elasticity.html"},"34":{"t":"Evolution problemsSymbole","p":"","l":"model/evolutionProblems.html"},"35":{"t":"Fluid-structure coupled problemSymbole","p":"","l":"model/fluidStructureCoupledProblem.html"},"36":{"t":"Free boundary problemsSymbole","p":"","l":"model/freeBoundaryProblem.html"},"37":{"t":"Mathematical ModelsSymbole","p":"","l":"model/index.html"},"38":{"t":"Navier-Stokes equationsSymbole","p":"","l":"model/navierStokesEquations.html"},"39":{"t":"Non-linear elasticitySymbole","p":"","l":"model/nonLinearElasticity.html"},"40":{"t":"Non-linear static problemsSymbole","p":"","l":"model/nonLinearStaticProblems.html"},"41":{"t":"Static problemsSymbole","p":"","l":"model/staticProblems.html"},"42":{"t":"Transmission problemSymbole","p":"","l":"model/transmissionProblem.html"},"43":{"t":"Variational InequalitySymbole","p":"","l":"model/variationalInequality.html"},"44":{"t":"Whispering gallery modesSymbole","p":"","l":"model/whisperingGalleryModes.html"},"45":{"t":"I/OSymbole","p":"","l":"reference/IO.html"},"46":{"t":"External librariesSymbole","p":"","l":"reference/externalLibraries.html"},"47":{"t":"FunctionsSymbole","p":"","l":"reference/functions.html"},"48":{"t":"Global variablesSymbole","p":"","l":"reference/globalVariables.html"},"49":{"t":"Language referencesSymbole","p":"","l":"reference/index.html"},"50":{"t":"LoopsSymbole","p":"","l":"reference/loops.html"},"51":{"t":"OperatorsSymbole","p":"","l":"reference/operators.html"},"52":{"t":"Quadrature formulaeSymbole","p":"","l":"reference/quadratureFormulae.html"},"53":{"t":"TypesSymbole","p":"","l":"reference/types.html"},"54":{"t":"ReferencesSymbole","p":"","l":"reference.html"},"55":{"t":"A Large Fluid ProblemSymbole","p":"","l":"tutorial/aLargeFluidProblem.html"},"56":{"t":"AcousticsSymbole","p":"","l":"tutorial/acoustics.html"},"57":{"t":"An Example with Complex NumbersSymbole","p":"","l":"tutorial/complexNumbers.html"},"58":{"t":"The System of elasticitySymbole","p":"","l":"tutorial/elasticity.html"},"59":{"t":"Classification of partial differential equationsSymbole","p":"","l":"tutorial/equationsClassification.html"},"60":{"t":"Irrotational Fan Blade Flow and Thermal effectsSymbole","p":"","l":"tutorial/fanBlade.html"},"61":{"t":"A Flow with ShocksSymbole","p":"","l":"tutorial/flowWithShocks.html"},"62":{"t":"Time dependent schema optimization for heat equationsSymbole","p":"","l":"tutorial/heatEquationOptimization.html"},"63":{"t":"Heat ExchangerSymbole","p":"","l":"tutorial/heatExchanger.html"},"64":{"t":"TutorialsSymbole","p":"","l":"tutorial/index.html"},"65":{"t":"Plotting in Matlab and OctaveSymbole","p":"","l":"tutorial/matlabOctavePlot.html"},"66":{"t":"MembraneSymbole","p":"","l":"tutorial/membrane.html"},"67":{"t":"Newton Method for the Steady Navier-Stokes equationsSymbole","p":"","l":"tutorial/navierStokesNewton.html"},"68":{"t":"A projection algorithm for the Navier-Stokes equationsSymbole","p":"","l":"tutorial/navierStokesProjection.html"},"69":{"t":"Optimal ControlSymbole","p":"","l":"tutorial/optimalControl.html"},"70":{"t":"Solving Poisson’s equationSymbole","p":"","l":"tutorial/poisson.html"},"71":{"t":"Pure Convection : The Rotating HillSymbole","p":"","l":"tutorial/rotatingHill.html"},"72":{"t":"The System of Stokes for FluidsSymbole","p":"","l":"tutorial/stokes.html"},"73":{"t":"Thermal ConductionSymbole","p":"","l":"tutorial/thermalConduction.html"},"74":{"t":"Tutorial to write a transient Stokes solver in matrix formSymbole","p":"","l":"tutorial/timeDependentStokes.html"},"75":{"t":"Wifi PropagationSymbole","p":"","l":"tutorial/wifiPropagation.html"}},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{},
{}];