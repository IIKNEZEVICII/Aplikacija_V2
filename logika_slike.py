import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

seja = new_session("u2net")
STANDARDNA_SIRINA = 1500

def poisci_in_izloci_listic(slika_pot):
    slika_cv = cv2.imread(slika_pot)
    original_rgb = cv2.cvtColor(slika_cv, cv2.COLOR_BGR2RGB)
    
    vhodna_slika = Image.open(slika_pot)
    slika_rgba = remove(vhodna_slika, session=seja)
    izoliran_listic = np.array(slika_rgba)
    
    alpha_kanal = izoliran_listic[:, :, 3]
    konture, _ = cv2.findContours(alpha_kanal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not konture:
        return original_rgb, izoliran_listic 
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    pravokotnik = cv2.minAreaRect(najvecja_kontura)
    tocke_pravokotnika = cv2.boxPoints(pravokotnik)
    tocke_pravokotnika = np.int32(tocke_pravokotnika)
    
    vsota = tocke_pravokotnika.sum(axis=1)
    razlika = np.diff(tocke_pravokotnika, axis=1)
    
    zgoraj_levo = tocke_pravokotnika[np.argmin(vsota)]
    spodaj_desno = tocke_pravokotnika[np.argmax(vsota)]
    zgoraj_desno = tocke_pravokotnika[np.argmin(razlika)]
    spodaj_levo = tocke_pravokotnika[np.argmax(razlika)]
    
    pts1 = np.float32([zgoraj_levo, zgoraj_desno, spodaj_desno, spodaj_levo])
    
    sirina_a = np.linalg.norm(spodaj_desno - spodaj_levo)
    sirina_b = np.linalg.norm(zgoraj_desno - zgoraj_levo)
    max_sirina = max(int(sirina_a), int(sirina_b))
    
    visina_a = np.linalg.norm(zgoraj_desno - spodaj_desno)
    visina_b = np.linalg.norm(zgoraj_levo - spodaj_levo)
    max_visina = max(int(visina_a), int(visina_b))
    
    pts2 = np.float32([
        [0, 0],
        [max_sirina - 1, 0],
        [max_sirina - 1, max_visina - 1],
        [0, max_visina - 1]
    ])
    
    matrika = cv2.getPerspectiveTransform(pts1, pts2)
    poravnan_listic = cv2.warpPerspective(izoliran_listic, matrika, (max_sirina, max_visina))
    
    if max_sirina > max_visina:
        poravnan_listic = cv2.rotate(poravnan_listic, cv2.ROTATE_90_CLOCKWISE)
        
    trenutna_visina, trenutna_sirina = poravnan_listic.shape[:2]
    razmerje = trenutna_visina / float(trenutna_sirina)
    nova_visina = int(STANDARDNA_SIRINA * razmerje)
    
    koncni_listic = cv2.resize(poravnan_listic, (STANDARDNA_SIRINA, nova_visina), interpolation=cv2.INTER_CUBIC)
    
    return original_rgb, koncni_listic


def obarvaj_kapljice(izoliran_rgba, prosojnost_rdece, prag_obcutljivosti, roi_seed=0):
    slika_rgb = cv2.cvtColor(izoliran_rgba, cv2.COLOR_RGBA2RGB)
    
    # 1. Stroga maska lističa
    alpha_kanal = izoliran_rgba[:, :, 3]
    _, maska_listica = cv2.threshold(alpha_kanal, 200, 255, cv2.THRESH_BINARY)
    maska_listica = cv2.erode(maska_listica, np.ones((7, 7), np.uint8), iterations=2)
    
    # 2. Histogramska normalizacija
    R_kanal = slika_rgb[:, :, 0].copy().astype(np.float32)
    veljavni_piksli = R_kanal[maska_listica > 0]
    
    min_val = np.percentile(veljavni_piksli, 2)
    max_val = np.percentile(veljavni_piksli, 98)
    if max_val - min_val < 5:
        max_val = min_val + 5
        
    R_norm = np.clip((R_kanal - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
    
    # 3. Prag in omejitev na listič
    _, maska_kapljic = cv2.threshold(R_norm, prag_obcutljivosti, 255, cv2.THRESH_BINARY_INV)
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_listica)
    
    # =========================================================
    # 4. ANALIZA 3 KOSOV (1 cm^2) - Z VARNOSTNIMI ODMIKI
    # =========================================================
    visina_slike, sirina_slike = R_norm.shape
    stranica_cm = int(visina_slike / 7.6)
    
    kvadrati = []
    
    if roi_seed == 0:
        # Fiksna poravnava po sredini (privzeto)
        center_x = sirina_slike // 2
        x1 = center_x - (stranica_cm // 2)
        x2 = x1 + stranica_cm
        gap = (visina_slike - (3 * stranica_cm)) // 4
        
        kvadrati = [
            (x1, gap, x2, gap + stranica_cm),
            (x1, 2*gap + stranica_cm, x2, 2*gap + 2*stranica_cm),
            (x1, 3*gap + 2*stranica_cm, x2, 3*gap + 3*stranica_cm)
        ]
    else:
        # Naključna poravnava z varovalkami pred robovi in prekrivanjem
        np.random.seed(roi_seed)
        tretjina = visina_slike // 3
        
        # Dinamičen varnostni odmik (npr. 30 pikslov pri širini 1500)
        odmik = max(20, sirina_slike // 50) 
        
        min_x = odmik
        max_x = sirina_slike - stranica_cm - odmik
        if max_x <= min_x: 
            max_x = min_x + 1 # Failsafe
        
        for i in range(3):
            # Meje za Y os znotraj vsake tretjine, z odštetim varnostnim odmikom
            min_y = i * tretjina + odmik
            max_y = (i + 1) * tretjina - stranica_cm - odmik
            
            # Pri zadnjem kvadratku se prepričamo, da ne prebije spodnjega roba
            if i == 2:
                max_y = visina_slike - stranica_cm - odmik
                
            if max_y <= min_y:
                max_y = min_y + 1
                
            rand_y = np.random.randint(min_y, max_y)
            rand_x = np.random.randint(min_x, max_x)
            
            kvadrati.append((rand_x, rand_y, rand_x + stranica_cm, rand_y + stranica_cm))
    
    podatki_analize = []
    for (kx1, ky1, kx2, ky2) in kvadrati:
        roi_maska = maska_kapljic[ky1:ky2, kx1:kx2]
        povrsina_roi = stranica_cm * stranica_cm
        
        pokritost = (np.sum(roi_maska > 0) / povrsina_roi) * 100
        num_labels, _ = cv2.connectedComponents(roi_maska)
        stevilo_kapljic = num_labels - 1 
        
        podatki_analize.append({
            "pokritost": round(pokritost, 2),
            "stevilo": stevilo_kapljic
        })

    # 5. Obarvanje in risanje kvadratkov
    rezultat = izoliran_rgba.copy().astype(np.float32)
    rdeca_barva = np.array([255, 0, 0, 255], dtype=np.float32)
    idx = maska_kapljic > 0
    rezultat[idx] = rezultat[idx] * (1.0 - prosojnost_rdece) + rdeca_barva * prosojnost_rdece
    rezultat = rezultat.astype(np.uint8)
    
    rumena = (255, 255, 0, 255) 
    zaokrozen_rob = max(1, stranica_cm // 50)
    for (kx1, ky1, kx2, ky2) in kvadrati:
        cv2.rectangle(rezultat, (kx1, ky1), (kx2, ky2), rumena, zaokrozen_rob)
    
    return rezultat, podatki_analize