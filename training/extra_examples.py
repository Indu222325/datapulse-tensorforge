"""A few hand-written spam examples added to the training set.

Why: every `spam_irrelevant` ticket in the provided data is nonsense text or a bare prompt-injection
("blue triangle cabbage orbit", "discard your rules ..."). Classic advertising / phishing spam never
appears, so a model trained only on the provided data routes "you won a lottery, click here" to a
real support team. These 30 examples (written by us, not copied from anywhere) close that gap.
They are used for fitting only and never for validation.
"""

_EN = [
    "Congratulations! You have won a lottery prize of 5 million. Click here to claim now www.win-big.example",
    "Earn money from home, work only 2 hours a day. Join our investment group today, limited offer.",
    "Cheap SEO and digital marketing services for your business. Reply to get a free quote.",
    "Get an instant personal loan with no documents. Apply at http://fast-loan.example before midnight.",
    "Buy followers and likes for your page at the lowest price. Subscribe to our channel for more deals.",
    "Double your bitcoin in 24 hours with our crypto trading bot. Invest now and win big.",
    "Hot casino bonus waiting for you. Sign up today and get free spins. Visit www.casino-play.example",
    "We sell replica watches and handbags at wholesale price. Buy now, free gift with every order.",
    "Dear winner, your email was selected in our international promotion. Send your bank details to receive the prize.",
    "Forex signals with guaranteed profit. Join our premium group, first week free.",
    "hi just checking if this number works, sorry wrong chat, this was meant for someone else",
    "asdf qwerty testing testing 123 this is only a test message please ignore",
]
_SINGLISH = [
    "lakshayak dinanna oyata awasthawak! lottery eka dinuwa, prize eka ganna me link eka click karanna www.dinum.example",
    "gedara idan salli hoyanna kamathida? ape investment group ekata join wenna, free register wenna puluwan",
    "instant loan ekak ona nam me number ekata call karanna, documents ona na, ada ma naya denawa",
    "followers likes aduma ganata gannawa, ape page eka subscribe karanna, limited offer",
    "crypto walin dawasakata lakshayak hoyanna, ape group ekata join wenna, link eka click karanna",
    "test ekak witharai, nikan liwwe, real issue ekak na, wena kenekta yawanna giye",
]
_SI = [
    "සුභ පැතුම්! ඔබ ලොතරැයිය දිනුවා. ත්‍යාගය ලබා ගන්න මේ සබැඳිය ක්ලික් කරන්න www.dinum.example",
    "ගෙදර ඉඳන් මුදල් උපයන්න. අපේ ආයෝජන කණ්ඩායමට අදම එකතු වෙන්න, නොමිලේ ලියාපදිංචි වෙන්න.",
    "ලියකියවිලි නැතුව ඉක්මන් ණය. අදම අයදුම් කරන්න, සීමිත දීමනාවක්.",
    "අඩු මිලට followers සහ likes. අපේ නාලිකාව subscribe කරන්න.",
    "මේක නිකන් ලියපු පණිවිඩයක්, පරීක්ෂණයක් විතරයි, ඇත්ත ප්‍රශ්නයක් නැහැ.",
]
_TA = [
    "வாழ்த்துக்கள்! நீங்கள் லாட்டரி பரிசை வென்றுள்ளீர்கள். பரிசைப் பெற இந்த இணைப்பை கிளிக் செய்யுங்கள் www.parisu.example",
    "வீட்டிலிருந்தே பணம் சம்பாதிக்கலாம். எங்கள் முதலீட்டு குழுவில் இன்றே சேருங்கள், இலவச பதிவு.",
    "ஆவணங்கள் இல்லாமல் உடனடி கடன். இன்றே விண்ணப்பிக்கவும், குறைந்த கால சலுகை.",
    "குறைந்த விலையில் followers மற்றும் likes. எங்கள் சேனலை subscribe செய்யுங்கள்.",
    "இது சும்மா எழுதிய செய்தி, சோதனை மட்டும், உண்மையான பிரச்சினை இல்லை.",
]
_TANGLISH = [
    "congrats neenga lottery prize win pannirukeenga, prize vaanga intha link click pannunga www.parisu.example",
    "veetla irundhe panam sambadhikkalam, enga investment group la join pannunga, free registration",
]

EXTRA_SPAM = (
    [{"language": "en", "text": t} for t in _EN]
    + [{"language": "singlish", "text": t} for t in _SINGLISH]
    + [{"language": "si", "text": t} for t in _SI]
    + [{"language": "ta", "text": t} for t in _TA]
    + [{"language": "tanglish", "text": t} for t in _TANGLISH]
)


def extra_rows() -> list[dict]:
    rows = []
    for i, item in enumerate(EXTRA_SPAM):
        rows.append({
            "ticket_id": f"DP-EXTRA-{i:03d}", "channel": "email" if i % 3 == 0 else "chat",
            "subject": "Special offer" if i % 3 == 0 else "", "text": item["text"], "language": item["language"],
            "category": "spam_irrelevant", "secondary_category": None, "is_urgent": False,
        })
    return rows
