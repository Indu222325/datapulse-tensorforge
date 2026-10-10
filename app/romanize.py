"""Script unification: Sinhala / Tamil Unicode -> rough Latin ("Singlish"-style) romanisation.

Why: 25% of tickets are Singlish (Sinhala typed in Latin letters) and 20% are Sinhala script.
They say the same things with the same words, but share no characters. After romanisation
"සල්ලි කැපිලා" becomes "salli kapila", which is exactly what a Singlish user types, so one set
of character n-gram features covers both scripts. Vowel length is dropped on purpose because
casual Singlish spelling does not mark it. Latin text passes through unchanged.
"""
SI_C = {"ක":"k","ඛ":"k","ග":"g","ඝ":"g","ඞ":"ng","ඟ":"ng","ච":"ch","ඡ":"ch","ජ":"j","ඣ":"j","ඤ":"ny","ඥ":"gn","ඦ":"nj","ට":"t","ඨ":"t","ඩ":"d","ඪ":"d","ණ":"n","ඬ":"nd","ත":"th","ථ":"th","ද":"d","ධ":"d","න":"n","ඳ":"nd","ප":"p","ඵ":"p","බ":"b","භ":"b","ම":"m","ඹ":"mb","ය":"y","ර":"r","ල":"l","ව":"w","ශ":"sh","ෂ":"sh","ස":"s","හ":"h","ළ":"l","ෆ":"f"}
SI_V = {"අ":"a","ආ":"a","ඇ":"a","ඈ":"a","ඉ":"i","ඊ":"i","උ":"u","ඌ":"u","එ":"e","ඒ":"e","ඓ":"ai","ඔ":"o","ඕ":"o","ඖ":"au","ඍ":"ru"}
SI_S = {"ා":"a","ැ":"a","ෑ":"a","ි":"i","ී":"i","ු":"u","ූ":"u","ෙ":"e","ේ":"e","ෛ":"ai","ො":"o","ෝ":"o","ෞ":"au","ෘ":"ru","ෲ":"ru"}
TA_C = {"க":"k","ங":"ng","ச":"s","ஜ":"j","ஞ":"ny","ட":"t","ண":"n","த":"th","ந":"n","ப":"p","ம":"m","ய":"y","ர":"r","ல":"l","வ":"v","ழ":"zh","ள":"l","ற":"r","ன":"n","ஷ":"sh","ஸ":"s","ஹ":"h"}
TA_V = {"அ":"a","ஆ":"a","இ":"i","ஈ":"i","உ":"u","ஊ":"u","எ":"e","ஏ":"e","ஐ":"ai","ஒ":"o","ஓ":"o","ஔ":"au"}
TA_S = {"ா":"a","ி":"i","ீ":"i","ு":"u","ூ":"u","ெ":"e","ே":"e","ை":"ai","ொ":"o","ோ":"o","ௌ":"au"}
CONS = {**SI_C, **TA_C}; VOW = {**SI_V, **TA_V}; SIGN = {**SI_S, **TA_S}
HAL = {"්", "்"}; SKIP = {"‍", "‌"}; ANUS = {"ං":"n", "ඃ":"h", "ஃ":"h"}
def romanize(text: str) -> str:
    """Return text with Sinhala/Tamil letters replaced by Latin approximations."""
    out = []; n = len(text); i = 0
    while i < n:
        ch = text[i]
        if ch in CONS:
            out.append(CONS[ch]); j = i + 1
            while j < n and text[j] in SKIP: j += 1
            if j < n and text[j] in HAL: i = j + 1; continue
            if j < n and text[j] in SIGN: out.append(SIGN[text[j]]); i = j + 1; continue
            out.append("a"); i = j; continue
        if ch in VOW: out.append(VOW[ch])
        elif ch in ANUS: out.append(ANUS[ch])
        elif ch in SKIP or ch in HAL or ch in SIGN: pass
        else: out.append(ch)
        i += 1
    return "".join(out)
