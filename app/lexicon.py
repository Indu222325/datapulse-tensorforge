"""Domain concept lexicon for RideEat support tickets.

Hand-written from general domain knowledge (ride-hailing + food delivery support),
NOT copied from dataset sentences. Each concept has regex alternations for:
  en : English (also catches English loan-words inside Singlish / Tanglish / mixed text)
  si : romanised Sinhala + Singlish
  ta : romanised Tamil + Tanglish
All patterns are matched on lower-cased, romanised text, anchored at a word start.
The classifier learns how much to trust each concept; the lexicon only says
"a word about <concept> is present", so it still fires on phrasings never seen in training.
"""
import re

LEX = {
 "pay_charge": {
  "en": r"charg\w*|overcharg\w*|bill(?:ed|ing)?\b|fees?\b|fare\b|deduct\w*|debit\w*|paid\b|pay(?:ing|ment|ments)?\b|transaction\w*|invoice\w*|cash\b|card\b|wallet|top-?up|balance|bank\b|transfer\w*|funds?\b|owe\b|settled?\b|statement|surcharge|toll\b|tariff|pric(?:e|ed|ing)\b|cost\b",
  "si": r"salli|mudal|gasthu|aya\s?k[ae]r|aya\s?kal|kap(?:ila|una|ena|uwa)|bil(?:a|e|ak)?\b|gew(?:im|wa|apu|anna|uwa)|ganak\b|gana\b|ganata|kad\b|wallet|balance|payment|charge|bank|ginumen",
  "ta": r"panam|kasu|kaasu|kattanam|pil\b|vasul|kazhi(?:kk|thth)|pitithth|pidich|attai|kartu|card|payment|charge|selavu|thokai|vangi|bank"},
 "pay_dup": {
  "en": r"twice|double|duplicat\w*|two (?:separate )?(?:charges|deductions|payments|times|debits)|second time|both (?:cash|card)|as well as (?:cash|card)",
  "si": r"depar|de\s?parak|wathawak\s?dek|dekama",
  "ta": r"irantu\s?murai|rendu\s?(?:thadava|murai|vaatti)|irumurai"},
 "refund": {
  "en": r"refund\w*|reimburs\w*|money back|compensat\w*|revers\w*|repay\w*|chargeback|credit(?:ed)? back|return(?:ed)? (?:my|the) (?:money|amount|payment)|give (?:me )?(?:back )?my money|promised to return",
  "si": r"refund|rifand|apasu|apahu|ayeth\s?denna|wandi",
  "ta": r"refund|rihpant|rifund|thirupp|thirump|thirumb|izhappit"},
 "credit": {"en": r"credit\w*", "si": r"credit|kredit", "ta": r"credit|kiretit"},
 "driver": {
  "en": r"driver\w*|drivr|chauffeur|captain\b|operator\b|person assigned",
  "si": r"riyadur|drayiwar|driver|draiwar",
  "ta": r"ottunar|tiraivar|driver|oattunar"},
 "vehicle": {
  "en": r"vehicle|car\b|tuk|three-?wheel\w*|bike\b|motorbike|cab\b|taxi|sedan|van\b|suv\b|hatchback|scooter",
  "si": r"wahana|car\b|kar\b|thriwil|thriroda|thunroda|three\s?wheel|bike|bayik|van\b",
  "ta": r"vanti|vakanam|atto|auto\b|kar\b|car\b|bike|paik|mussakkara"},
 "trip": {
  "en": r"rides?\b|trips?\b|journey|route|pick-?up|picked (?:me |us )?up|collect (?:me|us)|drop-?off|dropped|destination|detour|passenger|meter\b|street|location|longer way|long way",
  "si": r"ride|trip|gaman|para\b|paren|margaya|pickup|pick\s?up|drop|nawathana|yana\s?than|gamanantha|bassuwa|bahinna|stop\s?ek",
  "ta": r"ride|trip|payanam|savari|vazhi|pathai|pickup|drop|irakk|sellum\s?itam|iranku"},
 "courier": {
  "en": r"courier|rider\b|delivery (?:person|guy|man|partner|agent|driver|rider)|person bringing|deliverer",
  "si": r"rider|delivery\s?(?:kenek|karaya|eka\s?gena)|genena\s?kena",
  "ta": r"rider|delivery\s?(?:al|karar|boy)|kontuvarupavar"},
 "cancel": {
  "en": r"cancel\w*|withdrew|withdraw\w*|reject\w*|declin\w*|no-?show|never showed|did not show|didn'?t show|abandon\w*|called off|terminated",
  "si": r"cancel|kansal|awalangu|prathikshep|nawath\w*\s?dam",
  "ta": r"cancel|raththu|nirakari|kensal"},
 "lost": {
  "en": r"lost\b|left behind|forgot\w*|left (?:my|it|them)|misplac\w*|belongings|purse|backpack|handbag|laptop|keys?\b|umbrella|raincoat|charger|headphones?|earbuds|spectacles|glasses|passport|jacket|back ?seat|under the seat|suitcase|luggage|my (?:wallet|phone|bag)",
  "si": r"amathak|nathi\s?wun|nathiwun|dala\s?aw|dala\s?awith|pars\b|purse|yathur|kuday|laptop|asuna\s?yata|watila|sit\s?ek|thiyala\s?aw|athharala",
  "ta": r"maranth|tholainth|thavaravit|vittuvit|pars\b|savi|kutai|laptop|irukkai|kanamal\s?pon"},
 "belong": {
  "en": r"bag\b|phone\b|wallet\b",
  "si": r"bag\b|bagaya|fon\b|phone|durakathan|pasumbiya",
  "ta": r"pai\b|pon\b|phone|kaipesi|tholaipesi"},
 "missing": {
  "en": r"missing|incomplete|not included|left out|absent|short\b|shortage|fewer|without (?:its|the|my|a)\b|not (?:among|in|inside) the|never (?:received|got) (?:the|my)|only half|half the|didn'?t come with|did not come with",
  "si": r"aduyi|adu\b|aduwen|nathuwa|athule\s?na|dala\s?na|labune\s?na|thibune\s?na|madi\b",
  "ta": r"kanavillai|kanom|kuraiv|kurainth|illamal|varavillai|kitaikkavillai|vittupo"},
 "wrong": {
  "en": r"wrong\b|substitut\w*|instead of|different (?:item|dish|meal|order|food)|never (?:ordered|selected|chose|picked|asked)|didn'?t order|did not order|mix-?up|mixed up|swapped|someone else'?s|not what (?:i|we) ordered|incorrect|other (?:customer|person)'?s",
  "si": r"waradi|wena\s?(?:kamak|ekak|item|order)|wenas|wenath\s?kenek|order\s?kale\s?na",
  "ta": r"thavar|veru\s?(?:unavu|item|order|oru)|marri|marrik"},
 "fooditem": {
  "en": r"dish\w*|meal\w*|beverage|drinks?\b|dessert|combo|add-?on|side\b|portion\w*|items?\b|sauce|juice|burger|pizza|rice|kottu|contents|food\b|order\b|package\b|parcel|takeaway|dinner|lunch|breakfast|bread|soup|restaurant|kitchen",
  "si": r"kama|kem|kaama|parsal|parcel|order|anawum|drink|dessert|item|bima|bath\b|kottu|soup|restaurant|apanasal|awanhal",
  "ta": r"unavu|sappat|sapat|artar|order|parsal|parcel|pana|item|soru|unavakam|restaurant|hottal"},
 "delay": {
  "en": r"late\b|delay\w*|waiting|waited|hasn'?t (?:arrived|come|reached)|not arrived|has not (?:arrived|turned up|come|reached|shown)|still not|too long|hours?\b|minutes\b|not moving|stopped (?:moving|making)|overdue|yet to (?:arrive|reach|come)|stranded|queue|estimated|eta\b|arrival|still (?:waiting|preparing)|no sign|disappeared|where is my|behind schedule|not delivered|never (?:arrived|came|delivered)|parked|(?:same|one) (?:place|spot)|not here|not turned up",
  "si": r"late\b|parakku|pramada|thama\s?(?:na|awe|labun)|thawama|balagena|payak|paya\b|winadi|ekama\s?than|awe\s?na|labila\s?na|kal\s?giy|nawathila",
  "ta": r"late\b|let\b|thamatham|innum\s?var|kaththiru|mani\s?neram|nimitam|vanthataiya|vanthu\s?serav|ore\s?itath|varala|neramak"},
 "food_bad": {
  "en": r"cold\b|lukewarm|stale|spoil\w*|rotten|raw\b|undercook\w*|uncook\w*|smell\w*|odou?r|hair\b|insect|fly\b|cockroach|worm|maggot|tast\w*|soggy|spill\w*|spilt|leak\w*|crush\w*|squash\w*|burnt|expired|mou?ld\w*|fung\w*|foreign object|plastic|sour\b|seal\b|contaminat\w*|hygien\w*|melted|rancid|container|packag\w*|packing|spread across|sharp object|glass\b|stone\b",
  "si": r"sithala|seethala|narak|ganda|gandak|kunu|amu\b|konda|kes\b|krumi|kurumini|rasa\b|rasak|pilissun|leak|themila|podi\s?wela|packaging|idimun|pusba|pus\b|kal\s?ikuth|pichchila|weeduru|katu\b",
  "ta": r"kulir|ari\s?po|kettu|narram|vasanai|muti\b|pussi|puchi|vekatha|suvai|kasin|sinthi|nasung|packaging|pazhaiya|punjai|kannati"},
 "sick": {
  "en": r"sick|vomit\w*|nause\w*|stomach|food poisoning|allerg\w*|rash|swell\w*|swollen|throat|breath\w*|symptom\w*|diarr\w*|unwell|hives|itch\w*|dizzy|faint\w*|pain\b|choking|lips",
  "si": r"wamane|bada\s?(?:ridenawa|amaru)|asathmik|idimil|husma|ugura|karakil|ridenawa|thol\b|asanipa|leda",
  "ta": r"vanthi|vayir|ovvamai|vikkam|mussu|thontai|mayakkam|vali\b|uthatu|utalnilai"},
 "login": {
  "en": r"log-?in\w*|logged|log (?:in|into|out)|sign-?in|signin|sign in|otp|password|passcode|verification|verify|authenticat\w*|credentials|locked out|access (?:my|the) account|cannot access|one-time",
  "si": r"login|log\s?wenn|log\s?wen|log\s?wel|otp|murapada|password|thahawuru\s?keth|athul\s?wenn|ginumata\s?yanna",
  "ta": r"login|ulnuzha|otp|katavussol|password|sariparpp|nuzhaiya"},
 "account": {
  "en": r"account|profile|suspend\w*|deactivat\w*|blocked|banned|phone number|email address|registered (?:number|email)",
  "si": r"ginum|account|profile|block|suspend|thahanam|ankaya\s?wenas",
  "ta": r"kanakku|account|profile|block|muthakk|thatai"},
 "promo": {
  "en": r"promo\w*|coupon\w*|voucher\w*|discount\w*|code\b|offers?\b|referral|reward\w*|cashback|points\b|bonus|loyalty|redeem\w*",
  "si": r"promo|wattam|kupan|wawuchar|voucher|discount|code\b|kethaya|offer|referral|reward|thyaga",
  "ta": r"promo|thallupati|salukai|kuppan|voucher|discount|code\b|kuriyit|offer|referral|vekumathi"},
 "threat": {
  "en": r"unsafe|scared|afraid|frighten\w*|terrified|threat\w*|harass\w*|abus\w*|assault\w*|hit (?:me|us)|slap\w*|punch\w*|touch\w*|grop\w*|weapon|knife|follow(?:ing|ed) (?:me|us)|stalk\w*|inappropriat\w*|shout\w*|yell\w*|scream\w*|aggress\w*|violen\w*|fight\w*|grabb\w*|swear\w*|swore|insult\w*|molest\w*|intimidat\w*|extort\w*|menac\w*|lewd|obscene|vulgar|misbehav\w*|racist|demanding|doorway|my safety|personal comments|won'?t leave",
  "si": r"bayai|baya\b|bayak|tharjan|gahuw|gahann|athawar|hirihar|baninaw|bannaw|banin|asabya|lagata\s?awa|pasupasa|allanna|kadu|pihiya|arakshawa|anaraksh",
  "ta": r"payam|payama|miratt|atithth|atikk|thontharavu|thitt|thittin|pinthotar|kaththi|pathukapp|asinkam|thakatha"},
 "danger_drive": {
  "en": r"drunk|alcohol|intoxicat\w*|speeding|reckless|accident|collid\w*|collision|overturn\w*|red light|wrong side|lane\b|asleep|dozing|phone while driving|vehicle has crashed|car crashed|crashed (?:into|with)|too fast|slow down|rash driving",
  "si": r"bila\b|bimath|beela|wegayen|accident|anathur|happun|happil|peralun|nidimath|hemin\s?yanna",
  "ta": r"kutithth|kutipothai|pothai|vekamaka|accident|vipaththu|mothi|thunki|methuvaka"},
 "injury": {
  "en": r"police|injur\w*|hurt\b|bleed\w*|blood|wound\w*|ambulance|hospital|unconscious",
  "si": r"polis|thuwal|le\s?galan|le\b|ambulance|rohal|ispirithal|sihiya\s?na",
  "ta": r"polis|kayam|raththam|ambulance|maruththuvamanai|mayangi"},
 "trapped": {
  "en": r"not stopping|won'?t stop|will not stop|refus\w* to stop|let (?:me|us) out|get out|locked (?:me|us|inside|the doors)|trapped|kidnap\w*|wrong direction|unknown (?:road|place|area)|will not allow|won'?t allow|won'?t let",
  "si": r"nawaththanne\s?na|nawathanne\s?na|bahinna\s?denne\s?na|lock\s?kar|hira\s?wel|athule\b|wena\s?parak",
  "ta": r"niruththa\s?(?:villai|matt|maten)|iranka\s?vit|irangka\s?vit|pooti|sikki|veru\s?vazhi"},
 "app_ui": {
  "en": r"app\b|application|in-app|screen\w*|button|interface|page\b|menu\b|panel|window\b|tab\b|keyboard|pin\b|icon|viewer|toggle|checkbox|dropdown|field\b|control\b|chat\b",
  "si": r"app\b|ap\b|ap\s?ek|yedum|thiray|screen|boththam|button|sithiyam|map\b|menu|pin\b|chat",
  "ta": r"app\b|ap\b|seyali|thirai|screen|poththan|button|varaipatam|map\b|menu|pin\b|chat"},
 "app_fail": {
  "en": r"crash(?:es|ing)?\b|freez\w*|frozen|hangs?\b|hanging|stuck\b|loading|loads?\b|loaded|blank|bugs?\b|error\w*|glitch\w*|not working|doesn'?t work|does not work|unresponsive|render\w*|tap\w*|click\w*|exits?\b|closes?\b|quits?\b|no effect|nothing happens|fails? to|won'?t (?:open|load|save|submit|respond)|does not (?:open|load|save|respond)|will not (?:open|load|save|submit)|spinning|time(?:d)? ?out|lag\w*|jumps back|empty screen",
  "si": r"crash|krash|hira\s?wen|hirawel|wada\s?karanne\s?na|wada\s?na\b|lod\s?wenn|load\s?wenn|error|close\s?wen|wahenaw|open\s?wenn|ebuwata|mokuth\s?wenne\s?na|his\s?thir",
  "ta": r"crash|velai\s?seyya|load\s?aka|error|mutukirathu|thirakka\s?villai|uraink|aluththin|ethuvum\s?natakka"},
 "app_feat": {
  "en": r"notification\w*|alerts?\b|gps\b|search\w*|update\w*|version|install\w*|sync\w*|tracking|tracker|map\b|status\b",
  "si": r"danumdim|notification|gps|update|version|install|tracking|search|hoyann",
  "ta": r"arivipp|notification|gps|update|version|install|tracking|thetal"},
 "inquiry": {
  "en": r"how (?:do|can|to|does|long|much|many)|what (?:is|are|does)|do you|does rideeat|is it possible|is there|can i\b|could you (?:explain|tell)|information|info\b|register\w*|sign up|become a|partner\w*|onboard\w*|requirement\w*|documents?\b|working hours|opening hours|available in|service area|payout|eligib\w*|apply|rules\b|polic(?:y|ies)|explain|wonder\w*|curious|feedback|suggestion|compliment|guide me|not sure (?:what|how)|cannot describe|ahead of time|in advance",
  "si": r"kohomada|mokakda\s?widih|liyapadinch|register|thorathur|denagann|puluwanda|partner|riyadurek\s?wenn|kiyada\b|keeyada|monawada\s?ona|niyam|prathipaththi",
  "ta": r"eppati|enna\s?vazhi|pathivu|register|thakaval|therinth|mutiyuma|partner|evvalavu|vithimurai|ennenna"},
 "spam": {
  "en": r"win\b|winner|won\b|lottery|prize|click here|crypto\w*|bitcoin|invest\w*|loans?\b|seo\b|marketing|subscribe|free gift|congratulations|https?:|www\.|casino|earn\b|followers|limited offer|forex|buy now|work from home|cheap\b|sale\b",
  "si": r"dinann|dinum|thyagayak|lotharayi|naya\b|loan|crypto|click|link\b|https?:|www\.|nomile",
  "ta": r"verri|lattari|parisu|katan\b|loan|crypto|click|link\b|https?:|www\.|ilavasam"},
 "irrelevant": {
  "en": r"imaginary|made-?up|fictional|pretend\w*|nonsense|gibberish|just testing|test message|testing whether|only testing|nothing (?:else )?to (?:add|report|say)|no (?:real |actual )?(?:issue|problem|request)|no ticket|not a (?:real )?(?:request|complaint|issue)|random words|wrong (?:chat|number|recipient)|meant for (?:a|an|another|someone)",
  "si": r"kalpitha|manakkalpitha|imaginary|aththa\s?prashnayak\s?na|real\s?issue\s?ekak\s?na|wena\s?information\s?na|kiyanna\s?deyak\s?na|test\s?ekak|nikan\s?liww",
  "ta": r"karpanai|unmaiyana\s?pirassinai\s?illai|veru\s?thakaval\s?ill|solvatharku\s?ethuvum\s?ill|summa\s?ezhuthin|test\b"},
 "injection": {
  "en": r"(?:ignore|discard|disregard|override|forget|bypass) (?:all |your |the |any )?(?:previous |prior |above )?(?:instructions?|rules|classifier|prompt|labels?)|system prompt|label i command|print (?:its|your) internal|mark this (?:as )?urgent|classify this as|you must (?:label|mark|classify)",
  "si": r"instructions\s?amathaka|upades\s?amathaka|urgent\s?kiyala\s?mark|hadisi\s?kiyala\s?lakunu",
  "ta": r"vazhimuraikalai\s?maranthu|avasaram\s?enak\s?kuri|instructions\s?marandhu"},
 "now": {
  "en": r"right now|at this moment|immediately|emergency|urgent\w*|help now|as we speak|still inside|right away|asap|this instant|at once|need help now|currently",
  "si": r"danma|dan\s?help|hadisi|ikmanin|me\s?welawe|me\s?mohothe|thama\s?athule|wahama",
  "ta": r"utanati|avasaram|ippave|ippothe|intha\s?nimitam|utane"},
 "vulnerable": {
  "en": r"medicine|medication|insulin|inhaler|prescription|baby|infant|formula|toddler|child(?:ren)?\b|elderly|pregnant|diabet\w*|patient|oxygen|dose\b",
  "si": r"beheth|insulin|bab[ae]|ladaru|formula|daruw|wayasaka|gabini|diyawadiya|ledek|dose",
  "ta": r"marunthu|insulin|kuzhanthai|formula|muthiyavar|karppini|noyali|pappa"},
 "fraud": {
  "en": r"fraud\w*|unauthori[sz]\w*|hacked|not mine|someone else|stolen|scam\w*|without my (?:permission|consent|knowledge)|unknown transaction|i did not (?:make|authori[sz]e)|compromised|takeover",
  "si": r"mage\s?newei|awasarayak\s?nathuw|hack|wanchaw|hora|nodanuwathw",
  "ta": r"anumathi\s?illamal|hack|mosati|thirut|ennutaiyathu\s?alla|theriyamal"},
 "not_serious": {
  "en": r"no safety|not a safety|no symptoms|not asking for a refund|just an expression|only joking|just joking|no one (?:is|was) (?:hurt|threatened)|nobody is being|haha|lol\b|figure of speech|i am (?:safe|fine|okay) now|already (?:home|safe)|no emergency",
  "si": r"dan\s?mama\s?gedara|safe\b|wihiluwak|kisima\s?anathurak\s?na|api\s?kawe\s?na|kawe\s?na",
  "ta": r"safe\b|pathukappaka\s?irukk|vilaiyattukk|sappitavillai"},
 "past": {
  "en": r"yesterday|last (?:week|night|month)|days ago|earlier this week|previously|in the past|already resolved|was resolved",
  "si": r"iye\b|giya\s?sathiy|dawas\s?kipayakata\s?kalin|kalin\s?dawasak|wisandun",
  "ta": r"nerru|kazhintha\s?varam|natkalukku\s?mun|thirvu"},
 "also": {
  "en": r"also\b|separately|in addition|additionally|another\b|on top of|as well\b|besides|apart from|second (?:issue|problem)|other (?:issue|problem)|separate\b|unrelated",
  "si": r"ekata\s?amatharaw|amatharaw|wenama|thawa\s?prashn|e\s?wagema|anith|thawath",
  "ta": r"athumattumalla|thaniyaka|melum|innoru|kutave|athotu|athe\s?pol"},
}

CONCEPTS = sorted(LEX)
_PAT = {c: re.compile(r"(?<![a-z])(?:" + "|".join(LEX[c][l] for l in ("en", "si", "ta")) + ")") for c in CONCEPTS}
FEATURE_NAMES = [f"lex:{c}" for c in CONCEPTS]

def concept_counts(rom_text):
    """rom_text: lower-cased, romanised ticket text -> one clipped count (0..1) per concept."""
    return [min(len(_PAT[c].findall(rom_text)), 3) / 3.0 for c in CONCEPTS]


def concept_matches(rom_text, limit=3):
    """Concept -> the words that triggered it (for the demo's explanation panel only)."""
    found = {}
    for c in CONCEPTS:
        words = []
        for m in _PAT[c].finditer(rom_text):
            w = m.group(0).strip()
            if w and w not in words:
                words.append(w)
        if words:
            found[c] = words[:limit]
    return found
