##################################
# Built-in functions for Thiran
##################################

# Math functions implemented:
# ---------------------------
# sin cos tan
# ceil floor
# pow factorial mod 
# lcm gcd sqrt 
# radians degrees

# Math Constants:
# ---------------
#pi = 3.141592
#e = 2.718281
#tau = 6.283185

# Random Built-in functions implemented:
# --------------------------------------
# randint, choice, random

import error
from sys import exit

def வெளியேறு():
    exit(0)

def வகை(x):
    t = type(x)
    if t == int:
        return 'முழுஎண்'
    elif t == float:
        return 'புள்ளிஎண்'
    elif t == str:
        return 'வாக்கியம்'
    elif t == list:
        return 'பட்டியல்'
    else:
        return 'முழுஎண்ணோ புள்ளிஎண்ணோ வாக்கியமோ பட்டியலோ அல்ல!'

def முழுஎண்(num):
    try:
        return int(num)
    except:
        error.Error("முழுஎண்() - முழுஎண்ணாக மாற்ற இயலவில்லை", str(num))
        exit()

def புள்ளிஎண்(num):
    try:
        return float(num)
    except:
        error.Error("புள்ளிஎண்() - புள்ளிஎண்ணாக மாற்ற இயலவில்லை", str(num))
        exit()

def வாக்கியம்(inp):
    try:
        return str(inp)
    except:
        error.Error("வாக்கியம்() - வாக்கியமாக மாற்ற இயலவில்லை", str(inp))
        exit()

def round(num):
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "round()")
        exit()
    return round(num)

def get_pi():
    return 3.141592

def get_e():
    return 2.718281

def get_tau():
    return 6.283185


def ceil(num):
    import math
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "ceil()")
        exit()
    return math.ceil(num)

def floor(num):
    import math
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "floor()")
        exit()
    return math.floor(num)

def sin(num):
    import math
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "sin()")
        exit()
    return math.sin(num)

def cos(num):
    import math
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "cos()")
        exit()
    return math.cos(num)

def tan(num):
    import math
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "tan()")
        exit()
    return math.tan(num)

def power(a, b):
    import math
    if type(a) not in [int, float] or type(b) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "power()")
        exit()
    return a ** b

def factorial(num):
    import math
    if num == 0.0:
        num = 0
    if type(num) != int:
        error.Error("பொருந்தாத oவகை உள்ளீடு (எதிர்பார்த்தது முழுஎண்)", "factorial()")
        exit()
    if num < 0:
        error.Error("-ve மதிப்புகளுக்கு factorial வரையறுக்க முடியாது", "factorial()")
        exit()
    return math.factorial(num)

def mod(a, b):
    import math
    if type(a) not in [int, float] or type(b) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "mod()")
        exit()
    if b == 0 or b == 0.0:
        error.Error("கணிதப் பிழை (0 ஆல் வகுக்க முடியாது)", "mod()")
        exit()
    return math.fmod(a, b)

def gcd(a, b):
    import math
    if a == 0.0:
        a = 0
    if b == 0.0:
        b = 0
    if type(a) != int or type(b) != int:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண்)", "gcd()")
        exit()
    return math.gcd(a, b)

def lcm(a, b):
    import math
    if a == 0.0:
        a = 0
    if b == 0.0:
        b = 0
    if type(a) != int or type(b) != int:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண்)", "lcm()")
        exit()
    return math.lcm(a, b)

def sqrt(num):
    import math
    if type(num) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "sqrt()")
        exit()
    if num < 0:
        error.Error("sqrt()ல் உள்ளிட்ட எண் +ve ஆக இருக்க வேண்டும் ")
        exit()
    return math.sqrt(num)

def radians(degrees):
    import math
    if type(degrees) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "radians()")
        exit()
    return math.radian(degrees)

def degrees(radians):
    import math
    if type(radians) not in [int, float]:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண் அல்லது புள்ளிஎண்)", "degrees()")
        exit()
    return math.degrees(radians)


def randint(a, b):
    import random
    if a == 0.0:
        a = 0
    if b == 0.0:
        b = 0
    if type(a) != int or type(b) != int:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது முழுஎண்)", "randint()")
        exit()
    return random.randint(a, b)

def choice(choices):
    import random
    print(choices)
    if type(choices) != list:
        error.Error("பொருந்தாத வகை உள்ளீடு (எதிர்பார்த்தது பட்டியல்)", "choice")
        exit()
    if len(choices) == 0:
        error.Error("பட்டியலில் குறைந்தது 1 element இருக்க வேண்டும்", "choice()")
        exit()
    return random.choice(choices)

def random():
    import random
    # returns a random floating number between 0 and 1
    return random.random()
