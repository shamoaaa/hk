from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
import pickle
import base64
import hashlib


class AESModelEncryptor:
    def __init__(self, password=None, key=None):
        """
        初始化AES加密器
        :param password: 用户密码（可选）
        :param key: 直接提供密钥（可选）
        """
        if key is not None:
            self.key = key[:32].ljust(32, b'\0')  # 确保256位密钥
        elif password is not None:
            # 使用PBKDF2从密码派生密钥
            self.key = PBKDF2(password.encode(), b'salt_1234567890', dkLen=32)
        else:
            # 生成随机密钥
            self.key = get_random_bytes(32)

    def encrypt_model(self, model):
        """
        加密XGBoost模型
        :param model: 训练好的XGBoost模型
        :return: 加密后的数据字典
        """
        # 1. 序列化模型
        model_data = pickle.dumps(model)

        # 2. 生成随机nonce（初始化向量）
        nonce = get_random_bytes(16)

        # 3. 创建AES加密器（使用GCM模式）
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)

        # 4. 加密数据
        ciphertext, tag = cipher.encrypt_and_digest(model_data)

        # 5. 将所有数据打包
        encrypted_data = {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'tag': base64.b64encode(tag).decode('utf-8'),
            'key_hash': hashlib.sha256(self.key).hexdigest()  # 用于验证
        }

        return encrypted_data

    def decrypt_model(self, encrypted_data):
        """
        解密XGBoost模型
        :param encrypted_data: 加密的数据字典
        :return: 解密后的模型
        """
        # 1. 解码base64数据
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        tag = base64.b64decode(encrypted_data['tag'])

        # 2. 创建AES解密器
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)

        # 3. 验证并解密
        try:
            model_data = cipher.decrypt_and_verify(ciphertext, tag)
        except ValueError:
            raise ValueError("解密失败：数据可能被篡改或密钥错误")

        # 4. 反序列化模型
        model = pickle.loads(model_data)
        return model

    def save_encrypted_model(self, model, filepath):
        """
        保存加密模型到文件
        """
        encrypted_data = self.encrypt_model(model)

        # 保存到文件
        with open(filepath, 'w') as f:
            import json
            json.dump(encrypted_data, f)

        # print(f"模型已加密并保存到 {filepath}")

    def load_encrypted_model(self, filepath):
        """
        从文件加载并解密模型
        """
        # 从文件读取
        with open(filepath, 'r') as f:
            import json
            encrypted_data = json.load(f)

        # 解密模型
        model = self.decrypt_model(encrypted_data)
        return model