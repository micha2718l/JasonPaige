import time
time0 = time.time()

import numpy as np
from PIL import Image, ImageChops
import imageio


def normalize(n):
    n = np.array(n)
    n /= np.max(np.abs(n))
    n *= 255.0 / n.max()
    return n


def stack_normal(channels):
    return np.dstack(map(normalize, map(np.array, np.array(channels)))).astype("uint8")


def DoRotation(x, y, RotRad=0):
    RotMatrix = np.array(
        [[np.cos(RotRad), np.sin(RotRad)], [-np.sin(RotRad), np.cos(RotRad)]]
    )

    return np.einsum("ji, mni -> jmn", RotMatrix, np.dstack([x, y]))


pokeball = Image.open("pokeball.png")

w = 1280
h = 720
fps = 60
time = 10


X0, Y0 = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
# np.putmask(X0, mask > 0, pokeR * X0)
# np.putmask(Y0, mask > 0, pokeG * Y0)


# Y0 = Y0 * np.array(pokemask)[:,:,1]

i_max = fps * time
ims = []
for i in range(i_max):
    iN = i / i_max
    iNPi = iN * 2 * np.pi
    iNS = (np.sin(iNPi) + 1) / 2

    pokemask = Image.new("RGBA", (w, h), color="#00000000")
    pR = pokeball.rotate(iN*360*time)
    #pR = pR.resize((int(h/1.5), int(h/1.5)))
    pos = (int((w/2) - (pR.width/2)), int((h/2) - (pR.height/2)))
    pokemask.paste(pR, pos)
    mask = np.array(pokemask)[:, :, 3]
    pokeR = np.array(pokemask)[:, :, 0]
    pokeG = np.array(pokemask)[:, :, 1]
    pokeB = np.array(pokemask)[:, :, 2]



    if i % fps == 0:
        print(i)
    X, Y = X0*iN*2, Y0*iN*2
    C = (X + Y * 1j) * 1
    C0 = np.power(C, 2) + C
    for ii in range(20):
        C0 = np.power(C0, 2) + C
        # print(C0.max())
        C0[np.abs(C0) > 30] = 0

    C0 = normalize(C0)
    R = np.sin(np.abs(C0) + 1 * (2 * np.pi / 3) + iNS * 2 * (2 * np.pi / 3))
    G = np.sin(np.abs(C0) + 2 * (2 * np.pi / 3) + iNS * 2 * (2 * np.pi / 3))
    B = np.sin(np.abs(C0) + 3 * (2 * np.pi / 3) + iNS * 2 * (2 * np.pi / 3))
    np.putmask(R, mask > 0, np.sin(pokeR) + R)
    np.putmask(G, mask > 0, np.sin(pokeG) + G)
    np.putmask(B, mask > 0, np.sin(pokeB) + B)
    # import code;code.interact(local=locals())
    im = Image.fromarray(stack_normal([R, G, B]), "RGB")
    #im.show(); exit()
    ims.append(im.convert("RGB"))
imageio.mimwrite("testFColorRotate2.avi", ims, fps=fps)
""" 
        i_max = fps * time
        ims = []
        for i in range(i_max):
            pokeball = pokeball.rotate(5)
            if (i % fps == 0): print(i)
            iN = (i / i_max)
            im = self.make_image(iN, self.X, self.Y, pokeball)
            ims.append(im.convert('RGB'))
            im.convert('RGB').save(f'frames/aajp_0_{i}.png')
        imageio.mimwrite('test3.avi', ims, fps=fps)

 """

# im
# Image.open('test.tiff')
# imP = Image.fromarray(HSV(H*0, S*0, V*0), 'RGB')
# imP.paste(pokeball, (0,0), pokeball)
# Hp = np.array(imP)[:,:,0]
# impp = Image.fromarray(HSV(np.array(Hp)/255.0, S*0.5, V*1.0), 'HSV')
# impp.convert('RGB')

print(time.time() - time0)