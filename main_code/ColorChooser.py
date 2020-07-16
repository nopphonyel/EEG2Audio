import random

DBG = True


def random_color_on_lum(lum):
    Wr = 0.2126
    Wg = 0.7152
    Wb = 0.0722
    if DBG:
        print("LUM=%.2F" % (lum), end='\t')
    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = (lum - (r * Wr) - (g * Wg)) / Wb
        if 0 <= b <= 255:
            break
    b = int(b)
    print("#%02X%02X%02X" % (r, g, b))
    return r, g, b


def gen_color(contrast, higher_lum):
    assert 0 <= higher_lum <= 255, "<X>: higher_lum should be in range 0 to 255"
    lower_lum = (higher_lum + 0.05 - (contrast * 0.05)) / contrast
    assert lower_lum >= 0, "<X>: lower luminance value lower than 0\n\tConsider to increase higher_lum " \
                           "or decrease contrast"
    random_color_on_lum(higher_lum)
    random_color_on_lum(lower_lum)


for i in range(50):
    gen_color(100, 160)
