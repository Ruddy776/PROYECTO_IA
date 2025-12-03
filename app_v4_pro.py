import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# CONFIGURACIÓN
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Las 10 Clases que conocemos
CLASES = {
    0: '✈️ Avión', 1: '🚗 Auto', 2: '🐦 Pájaro', 3: '🐱 Gato', 4: '🦌 Ciervo',
    5: '🐶 Perro', 6: '🐸 Rana', 7: '🐴 Caballo', 8: '🚢 Barco', 9: '🚛 Camión'
}

# UMBRAL DE CERTEZA (El filtro de calidad)
# Si la IA no está segura al menos en este porcentaje, dirá "Otro"
UMBRAL_CONFIANZA = 0.60  # 60%

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neural Vision AI - V4 Expert")
        self.geometry("1000x750")
        
        self.modelo = None
        self.cargar_modelo_ia()

        # --- LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Panel Izquierdo
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.sidebar, text="🧠 AI EXPERT", font=ctk.CTkFont(size=26, weight="bold")).grid(row=0, column=0, padx=20, pady=40)
        
        self.btn_subir = ctk.CTkButton(self.sidebar, text="📷 ANALIZAR FOTO", command=self.seleccionar_imagen, 
                                       height=50, font=ctk.CTkFont(size=16, weight="bold"), fg_color="#7c3aed") # Color Violeta
        self.btn_subir.grid(row=1, column=0, padx=20, pady=20)
        
        self.lbl_info = ctk.CTkLabel(self.sidebar, text=f"Filtro de Seguridad:\nMin. {int(UMBRAL_CONFIANZA*100)}% de certeza", text_color="gray")
        self.lbl_info.grid(row=2, column=0, padx=20, pady=10)

        # Panel Derecho
        self.main_area = ctk.CTkFrame(self, corner_radius=15)
        self.main_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.lbl_imagen = ctk.CTkLabel(self.main_area, text="Sube una imagen...", font=ctk.CTkFont(size=20), width=600, height=450)
        self.lbl_imagen.pack(pady=30, padx=20, expand=True)

        # Resultados
        self.frame_resultados = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.frame_resultados.pack(fill="x", padx=40, pady=(0, 30))

        self.lbl_ganador = ctk.CTkLabel(self.frame_resultados, text="", font=ctk.CTkFont(size=32, weight="bold"))
        self.lbl_ganador.pack(pady=5)
        
        self.lbl_subtexto = ctk.CTkLabel(self.frame_resultados, text="", font=ctk.CTkFont(size=16))
        self.lbl_subtexto.pack(pady=(0, 15))

        # Barra única de confianza
        self.barra_progreso = ctk.CTkProgressBar(self.frame_resultados, height=25, width=400)
        self.barra_progreso.pack()
        self.barra_progreso.set(0)

        self.verificar_modelo()

    def cargar_modelo_ia(self):
        try:
            if os.path.exists('modelo_v4_expert.keras'):
                self.modelo = tf.keras.models.load_model('modelo_v4_expert.keras')
                self.modelo_ok = True
            else:
                self.modelo_ok = False
        except:
            self.modelo_ok = False

    def verificar_modelo(self):
        if not self.modelo_ok:
            messagebox.showerror("Error", "¡Falta el cerebro V4!\nEjecuta 'entrenar_v4_pro.py' primero.")

    def seleccionar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg;*.png;*.jpeg")])
        if not path: return

        # Mostrar Imagen
        img_original = Image.open(path)
        base_height = 450
        w_percent = (base_height / float(img_original.size[1]))
        w_size = int((float(img_original.size[0]) * float(w_percent)))
        img_ctk = ctk.CTkImage(light_image=img_original, dark_image=img_original, size=(w_size, base_height))
        self.lbl_imagen.configure(image=img_ctk, text="")
        
        self.analizar(img_original)

    def analizar(self, img_pil):
        if not self.modelo: return

        # Preprocesar
        img_small = img_pil.resize((32, 32))
        img_arr = np.array(img_small)
        if img_arr.shape == (32, 32, 4): img_arr = img_arr[:,:,:3]
        if len(img_arr.shape) == 2: img_arr = np.stack((img_arr,)*3, axis=-1)
        input_data = img_arr.reshape(1, 32, 32, 3) / 255.0

        # Predicción
        pred = self.modelo.predict(input_data)[0]
        
        idx_ganador = np.argmax(pred)
        confianza = pred[idx_ganador]
        
        # --- LOGICA DEL UMBRAL (AQUÍ ESTÁ EL TRUCO) ---
        self.barra_progreso.set(confianza)
        
        if confianza < UMBRAL_CONFIANZA:
            # Caso: La IA está confundida (ej. es un niño o una caja)
            self.lbl_ganador.configure(text="❓ OBJETO DESCONOCIDO", text_color="#9ca3af") # Gris
            self.lbl_subtexto.configure(text=f"No coincide con mis 10 categorías (Certeza baja: {confianza*100:.1f}%)")
            self.barra_progreso.configure(progress_color="gray")
        else:
            # Caso: Sí sabe qué es
            nombre = CLASES[idx_ganador]
            self.lbl_ganador.configure(text=f"{nombre.upper()}", text_color="#4ade80") # Verde
            self.lbl_subtexto.configure(text=f"Identificado con {confianza*100:.1f}% de seguridad")
            self.barra_progreso.configure(progress_color="#4ade80")

if __name__ == "__main__":
    app = App()
    app.mainloop()