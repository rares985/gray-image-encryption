"""
Application entry point.
"""
from copy import deepcopy
import cv2
from encrypt import encrypt
from decrypt import decrypt
from utils import generate_key, MASTER_KEY

def main():

    # key_enc = generate_key()
    # key_enc = '02296fc04d63f52ca6966b02238ed8fe'
    key_enc=MASTER_KEY
    print(key_enc)

    image_name = 'lena.png'

    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    encrypted_img = encrypt(img, key_enc, permute=False, rounds=4)

    decrypted_img = decrypt(encrypted_img, key_enc, permute=False, rounds=4)

    cv2.imwrite('encrypted.png', encrypted_img)

    cv2.imwrite('decrypted.png', decrypted_img)

    return

if __name__ == "__main__":
    main()